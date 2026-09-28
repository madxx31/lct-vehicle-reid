"""Finetune C-RADIOv4 with ArcFace on data/train.csv.

    python -m train.encoder.train "data.eval_folds=[1,2]"  # oof embeddings -> oof_preds/embeddings12oof.npy
    python -m train.encoder.train                          # all data, weights -> models/vehicle_encoder.pt

Any config.yaml key can be overridden the same way, e.g. data.num_workers=8.
"""

import os

# reproducibility
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from omegaconf import open_dict, OmegaConf
from .data_module import FalconDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar, EMAWeightAveraging
from .model import ArcFaceModel
import logging
import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

ROOT = Path(__file__).parents[2]


def main() -> None:
    cfg = OmegaConf.merge(OmegaConf.load(Path(__file__).parent / "config.yaml"), OmegaConf.from_cli())
    seed_everything(cfg.seed, workers=True)
    dm = FalconDataModule(cfg)
    dm.setup()
    with open_dict(cfg):
        cfg.model.num_classes = dm.num_classes
    model = ArcFaceModel(cfg)
    ema = EMAWeightAveraging(decay=cfg.model.ema_decay)
    trainer = Trainer(**cfg.trainer, callbacks=[RichProgressBar(), ema])

    device = torch.device("cuda" if cfg.trainer.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    logging.info("seeding the ArcFace head with the pretrained encoder's class-mean embeddings")
    seeded = model.init_arcface_from_class_means(dm.train_noaugm_dataloader(), device)
    logging.info(f"seeded {seeded}/{dm.num_classes} class vectors")

    trainer.fit(model, dm)
    ema_model = ema._average_model.module

    if not cfg.data.eval_folds:
        out_path = ROOT / "models" / "vehicle_encoder.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ema_model.encoder.state_dict().items()}
        torch.save(state, out_path)
        logging.info(f"saved weights to {out_path}")
        return

    ema_model.eval()
    embeddings = []
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.inference_mode(), autocast:
        for batch in dm.val_dataloader():
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings.append(ema_model(batch).float().cpu())
    embeddings = F.normalize(torch.cat(embeddings, dim=0)).numpy()
    # embeddings12oof.npy for data.eval_folds=[1,2]
    out_path = ROOT / "oof_preds" / f"embeddings{''.join(map(str, cfg.data.eval_folds))}oof.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    logging.info(f"saved oof embeddings to {out_path}")


if __name__ == "__main__":
    main()
