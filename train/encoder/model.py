import torch
from collections import defaultdict
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import logging
import math
import re
from transformers import AutoModel
import transformers.dynamic_module_utils as dynamic_module_utils

# C-RADIO's remote code imports `open_clip` lazily, inside its CLIP text adaptor, which is only
# built when adaptors are requested. transformers' import check greps every file regardless and
# refuses to load without the package, so drop it from that check; the summary path never needs it.
_get_imports = dynamic_module_utils.get_imports
dynamic_module_utils.get_imports = lambda filename: [m for m in _get_imports(filename) if m != "open_clip"]


def param_depth(name, num_layers, layer_re, embed_marker):
    """Depth of an encoder parameter: 0 for the patch/position embeddings, 1..L for the transformer
    blocks, L+1 for everything above them.
    """
    match = layer_re.search(name)
    if match:
        return int(match.group(1)) + 1
    return 0 if embed_marker in name else num_layers + 1


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        margin=0.5,
        scale=30.0,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

        self.ls_eps = ls_eps
        self.arc_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.arc_weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input.float()), F.normalize(self.arc_weight.float()))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)), 1e-9, 1))

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class RadioEncoder(nn.Module):
    """NVIDIA's C-RADIOv4 (`nvidia/C-RADIOv4-SO400M`), loaded through its transformers remote code.

    It takes [0, 1] pixels at a fixed (height, width) and normalises them with its own input
    conditioner. The embedding is its summary output: the backbone's teacher-matched cls tokens,
    concatenated (2 x 1152 for SO400M). It is not a re-ID model, so the ArcFace head starts from scratch.
    The final L2 is left to the caller: ArcFace normalises its input and so does the retrieval metric.
    """

    input_keys = ("pixel_values",)
    # C-RADIO's timm ViT blocks: `radio_model.model.blocks.<i>.`, 0-based
    layer_re = re.compile(r"model\.blocks\.(\d+)\.")
    # patch embedding, CPE position embeddings, cls/register tokens
    embed_marker = "patch_generator."

    def __init__(self, hf_repo, revision, image_size, gradient_checkpointing=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_repo, revision=revision, trust_remote_code=True).float()
        radio = self.backbone.radio_model
        h, w = image_size
        nearest = tuple(radio.get_nearest_supported_resolution(h, w))
        assert nearest == (h, w), f"{h}x{w} (h x w) is not a supported RADIO input size, nearest is {nearest}"
        if gradient_checkpointing:
            # the timm ViT inside RADIO; its CPE forward runs `checkpoint_seq(self.blocks, x)` when this is set
            radio.model.set_grad_checkpointing(True)
        with torch.no_grad():
            was_training = self.backbone.training
            self.backbone.eval()
            self.out_dim = self.forward(torch.zeros(1, 3, h, w)).shape[-1]
            self.backbone.train(was_training)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values)
        return out.summary if hasattr(out, "summary") else out[0]

    def position_embedding(self):
        # in training mode RADIO's CPE resamples this table with a random crop through `grid_sample`,
        # whose CUDA backward has no deterministic kernel
        return self.backbone.radio_model.model.patch_generator.pos_embed


class ArcFaceModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        m = cfg.model
        self.encoder = RadioEncoder(m.hf_repo, m.revision, m.image_size, m.gradient_checkpointing)
        if m.freeze_position_embedding:
            # RADIO resamples this table with a random-crop `grid_sample`, and that op's CUDA backward
            # accumulates with atomics: with it trainable the run cannot be bit-for-bit reproducible. It
            # sits at the bottom of the layer-wise decay (~1.6e-6) and is all but frozen anyway, so
            # freezing it buys determinism cheaply.
            self.encoder.position_embedding().requires_grad_(False)
        assert self.encoder.out_dim == cfg.model.dim, f"encoder is {self.encoder.out_dim}-d, config says {cfg.model.dim}"
        self.arcface = ArcMarginProduct(
            self.encoder.out_dim,
            cfg.model.num_classes,
            margin=cfg.model.arcface.margin,
            scale=cfg.model.arcface.scale,
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.encoder(*(batch[k] for k in self.encoder.input_keys))

    @torch.no_grad()
    def init_arcface_from_class_means(self, dataloader, device):
        """Start each ArcFace class vector at the pretrained encoder's mean embedding for that class.

        Xavier-random class vectors would make the first steps push the encoder towards noise;
        class centres put the head where a well-trained one would already roughly be.
        """
        was_training = self.training
        self.to(device).eval()
        sums = torch.zeros_like(self.arcface.arc_weight)
        counts = torch.zeros(self.arcface.out_features, device=device)
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                emb = self(batch)
            sums.index_add_(0, batch["labels"], F.normalize(emb.float()))
            counts.index_add_(0, batch["labels"], torch.ones_like(batch["labels"], dtype=counts.dtype))
        seen = (counts > 0).unsqueeze(1)
        centres = F.normalize(sums / counts.clamp(min=1).unsqueeze(1))
        self.arcface.arc_weight.data = torch.where(seen, centres, self.arcface.arc_weight.data)
        # Lightning's validation loop captures and restores the current mode, so a module left in
        # eval mode here would stay in eval mode for the whole of fit()
        self.train(was_training)
        return int(seen.sum())

    def training_step(self, batch, batch_idx):
        embedded = self.forward(batch)
        preds = self.arcface(embedded, batch["labels"])
        loss = self.loss(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """One parameter group per encoder depth, with a layer-wise decayed learning rate.

        The pretrained lower blocks hold general features that ~1200 vehicles can only damage, while
        the top is the task-specific part worth moving fast. Depth d gets
        `learning_rate * top_mult * decay ** (L + 1 - d)`: the top runs `top_mult`x the base rate
        and the patch embeddings are all but frozen. The ArcFace head keeps its own absolute rate.
        """
        lr = self.cfg.model.learning_rate
        decay, top_mult = self.cfg.model.llrd.decay, self.cfg.model.llrd.top_mult
        layer_re, embed_marker = self.encoder.layer_re, self.encoder.embed_marker
        depths = {n: param_depth(n, 0, layer_re, embed_marker) for n, _ in self.encoder.named_parameters()}
        num_layers = max(depths.values())
        assert num_layers > 0, f"no transformer blocks matched {layer_re.pattern} in {list(depths)[:5]}"

        groups = {}
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            depth = param_depth(name, num_layers, layer_re, embed_marker)
            group = groups.setdefault(depth, {"params": [], "lr": lr * top_mult * decay ** (num_layers + 1 - depth)})
            group["params"].append(param)
        logging.info(
            "layer-wise lr decay over %d blocks: embeddings %.2e, top %.2e",
            num_layers,
            groups[0]["lr"],
            # RADIO has nothing above its last block, so the top group is not always L + 1
            groups[max(groups)]["lr"],
        )

        head = {"params": [self.arcface.arc_weight], "lr": lr * self.cfg.model.arcface.lr_mult, "weight_decay": 0}
        return AdamW(
            [head, *(groups[d] for d in sorted(groups))],
            lr=lr,
            weight_decay=self.cfg.model.weight_decay,
        )
