import math
import random
import torch
import pytorch_lightning as pl
from torchvision.transforms.v2 import functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import v2, InterpolationMode
from pathlib import Path
import logging

DATA_DIR = Path(__file__).parents[2] / "data"


class RelativeRandomCrop(torch.nn.Module):
    """Random area/aspect crop, left at its own resolution: it only has to choose the region."""

    def __init__(self, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3)):
        super().__init__()
        self.scale = scale
        self.log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    def forward(self, img):
        h, w = img.shape[-2:]
        area = random.uniform(*self.scale)
        r = math.exp(random.uniform(*self.log_ratio))
        cw = min(w, max(1, round(w * math.sqrt(area * r))))
        ch = min(h, max(1, round(h * math.sqrt(area / r))))
        top, left = random.randint(0, h - ch), random.randint(0, w - cw)
        return F.crop(img, top, left, ch, cw)


class FixedSizeTransform:
    """Crop at the image's own resolution, resize to the fixed (height, width) the model takes, then augment.

    The crop is squashed to `image_size`, which is why it is landscape, near the crops' median aspect
    ratio. Pixels stay in [0, 1]: RADIO's input conditioner normalises them itself. Perspective and
    colour jitter run after the resize, where they are much cheaper.
    """

    def __init__(self, image_size, train):
        self.image_size = list(image_size)
        self.crop = RelativeRandomCrop(scale=(0.6, 1.0)) if train else v2.Identity()
        post = []
        if train:
            post += [
                v2.RandomPerspective(distortion_scale=0.2, p=0.3),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        post.append(v2.ToDtype(torch.float32, scale=True))
        if train:
            post.append(v2.RandomErasing(p=0.5, value=0.5))  # mid-grey
        self.post = v2.Compose(post)

    def __call__(self, img):
        img = F.resize(self.crop(img), self.image_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        return {"pixel_values": self.post(img)}


class FalconDataset(Dataset):
    def __init__(self, df, img_dir, transform, labels=None):
        self.image_ids = df["image_id"].tolist()
        self.bboxes = list(df[["x", "y", "w", "h"]].itertuples(index=False, name=None))
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.labels = labels

    def crop(self, img, box):
        """Cut the annotated BBox out of the full frame, still uint8 CHW."""
        x, y, w, h = box
        ih, iw = img.shape[-2:]
        # the annotations are trusted but not guaranteed to sit inside the frame
        x, y = max(0, min(x, iw - 1)), max(0, min(y, ih - 1))
        return img[:, y : min(y + h, ih), x : min(x + w, iw)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = decode_jpeg(read_file(str(self.img_dir / f"{self.image_ids[idx]}.jpg")))
        img = self.crop(img, self.bboxes[idx])
        res = {**self.transform(img)}
        if self.labels is not None:
            res["label"] = self.labels[idx]
        return res


class FalconDataModule(pl.LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.already_setup = False

    def setup(self, stage=None):
        if self.already_setup:
            return
        img_dir = DATA_DIR / "images"

        df = pd.read_csv(DATA_DIR / "train.csv")
        # an empty fold list selects nothing, so everything lands in train
        is_val = (df["vehicle_id"] % 5).isin(self.cfg.data.eval_folds)
        tr_df = df[~is_val].reset_index(drop=True)
        val_df = df[is_val].reset_index(drop=True)

        tr_labels = pd.factorize(tr_df["vehicle_id"])[0]
        val_labels = pd.factorize(val_df["vehicle_id"])[0]
        self.num_classes = int(tr_labels.max()) + 1

        train_tf = FixedSizeTransform(self.cfg.model.image_size, train=True)
        eval_tf = FixedSizeTransform(self.cfg.model.image_size, train=False)
        self.tr_ds = FalconDataset(tr_df, img_dir, train_tf, tr_labels)
        # the same training images, unaugmented: used to seed the ArcFace head with class centres
        self.tr_noaugm_ds = FalconDataset(tr_df, img_dir, eval_tf, tr_labels)
        # out-of-fold images, in train.csv order: train/confidence_model.py relies on it
        self.val_ds = FalconDataset(val_df, img_dir, eval_tf, val_labels)

        logging.info(
            f"oof folds {list(self.cfg.data.eval_folds) or 'none'}: train {len(tr_df)} images / {self.num_classes} vehicles, "
            f"oof {len(val_df)} images / {val_df['vehicle_id'].nunique()} vehicles"
        )
        self.already_setup = True

    def train_dataloader(self):
        return DataLoader(
            self.tr_ds,
            batch_size=self.cfg.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collator,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True,
            # python 3.14 defaults to forkserver, which fails with "too many fds" when val workers start
            multiprocessing_context="fork",
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_ds)

    def train_noaugm_dataloader(self):
        return self._eval_dataloader(self.tr_noaugm_ds)

    def _eval_dataloader(self, ds):
        return DataLoader(
            ds,
            batch_size=self.cfg.data.eval_batch_size,
            drop_last=False,
            collate_fn=self.collator,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=False,
            multiprocessing_context="fork",
        )

    def collator(self, batch):
        res = {k: torch.stack([i[k] for i in batch]) for k in batch[0] if k != "label"}
        if "label" in batch[0]:
            res["labels"] = torch.tensor([i["label"] for i in batch], dtype=torch.long)
        return res
