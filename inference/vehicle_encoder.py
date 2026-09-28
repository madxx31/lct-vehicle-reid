"""The inference entry point a measurement harness imports.

    model = VehicleEmbedder()          # weights load here, outside the timed cycle
    emb = model.extract(items)         # (n, 2304) float32, L2-normalised, on the CPU

`items` is a list of {"image_path": "<frame>.jpg", "bbox": [x, y, w, h]}
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from train.encoder.data_module import FixedSizeTransform
from train.encoder.model import RadioEncoder

ROOT = Path(__file__).parents[1]
# the backbone, its revision and the input size the checkpoint was trained with
CONFIG = ROOT / "train" / "encoder" / "config.yaml"
# baked into the Docker image here; in a repo checkout it is what train/train.sh writes
IMAGE_CHECKPOINT = Path("/opt/model/vehicle_encoder.pt")
IN_IMAGE = IMAGE_CHECKPOINT.exists()
DEFAULT_CHECKPOINT = IMAGE_CHECKPOINT if IN_IMAGE else ROOT / "models" / "vehicle_encoder.pt"


class VehicleEmbedder:

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cuda", num_threads=32):
        self.device = torch.device(device)
        # no silent CPU fallback: a driver older than torch's CUDA build shows up as "no CUDA", and on the
        # CPU the service would still run, just ~100x slower. Pass device="cpu" to run there on purpose
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA is not available to torch {torch.__version__} (built for CUDA {torch.version.cuda}): "
                "check the GPU is visible (docker --gpus all) and the NVIDIA driver supports this CUDA version"
            )
        cfg = OmegaConf.load(CONFIG).model
        self.image_size = tuple(cfg.image_size)
        self.transform = FixedSizeTransform(cfg.image_size, train=False)

        self.encoder = RadioEncoder(cfg.hf_repo, cfg.revision, cfg.image_size)
        # strict, so a checkpoint that is not this encoder fails here rather than embedding badly
        self.encoder.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        # the checkpoint is bf16; keeping the module in bf16 too spares autocast from re-casting every
        # weight on every forward. Autocast stays on for the ops it runs in fp32 (norms, softmax)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.encoder.to(self.device, dtype).eval()

        # -1 takes every core the process may run on; anything else is capped by that count
        available = os.process_cpu_count() or 1
        num_threads = available if num_threads < 0 else min(num_threads, available)
        self.pool = ThreadPoolExecutor(num_threads) if num_threads > 1 else None
        # torchvision's decode and the tensor ops all release the GIL, so the pool scales; each
        # worker stays single-threaded to keep the pool from fighting torch's own intra-op threads
        torch.set_num_threads(1)

    # ---- the cycle, split into stages so a profiler can charge each one separately ----

    def read(self, path):
        """Pull the raw JPEG bytes of the full frame off disk."""
        return read_file(str(path))

    def decode(self, data):
        return decode_jpeg(data, mode=ImageReadMode.RGB)

    def crop(self, img, box):
        """Cut the annotated BBox out of the full frame, still uint8 CHW."""
        x, y, w, h = (int(v) for v in box)
        ih, iw = img.shape[-2:]
        # the annotations are trusted but not guaranteed to sit inside the frame
        x, y = max(0, min(x, iw - 1)), max(0, min(y, ih - 1))
        return img[:, y : min(y + h, ih), x : min(x + w, iw)]

    def preprocess(self, img):
        """Resize to the fixed input size, [0, 1] pixels: RADIO normalises them itself."""
        return self.transform(img)["pixel_values"]

    def collate(self, samples):
        return torch.stack(samples).to(self.device, non_blocking=True)

    @torch.inference_mode()
    def forward(self, pixel_values):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            emb = self.encoder(pixel_values)
        # RadioEncoder deliberately leaves the final L2 to the caller
        return F.normalize(emb.float())

    def _prepare(self, item):
        """The whole CPU half of the cycle for one vehicle."""
        img = self.decode(self.read(item["image_path"]))
        return self.preprocess(self.crop(img, item["bbox"]))

    # ---- what the harness calls ----

    def extract(self, items):
        """items: a list of {"image_path", "bbox"} -> (n, 2304) float32 numpy, L2-normalised."""
        if self.pool is None or len(items) == 1:
            samples = [self._prepare(i) for i in items]
        else:
            samples = list(self.pool.map(self._prepare, items))
        return self.forward(self.collate(samples)).cpu().numpy()
