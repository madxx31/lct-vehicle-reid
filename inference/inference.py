"""Embed the query/gallery crops and write `embeddings.npy`, `submission.csv` + `candidates.csv`.

Run inside the container, which binds the data to /data and the results to /output:

    python -m inference.inference [--batch-size N] [--num-threads N] [--threshold P]
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from .vehicle_encoder import VehicleEmbedder

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

DATA_DIR = Path("/data")  # bind-mounted: test_query.csv, test_gallery.csv, images/<image_id>.jpg
OUTPUT_DIR = Path("/output")  # bind-mounted: embeddings.npy, submission.csv and candidates.csv land here
CHECKPOINT = Path("/opt/model/vehicle_encoder.pt")  # baked into the image
CONFIDENCE_MODEL = Path("/opt/model/confidence_model.cbm")  # baked in too, see train/confidence_model.py
# picked on oof folds 3-4 by train/confidence_model.py: the centre of the F1 plateau
THRESHOLD = 0.3858
TOP_K = 10


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def DBA(G):
    """Database-side augmentation: blend each gallery vector with its nearest neighbour."""
    s = G @ G.T
    np.fill_diagonal(s, -np.inf)
    i = s.argmax(1)
    sim1 = s[np.arange(len(G)), i]
    return normalize(G + np.where(sim1 > 0.5, sim1.clip(min=0) ** 2, 0)[:, None] * G[i])


def confidence_features(sim, sim_raw, top1):
    """Per-query features of the top-1 candidate: how close it is, and how far it stands out.

    `sim` is against the augmented gallery (what the ranking uses), `sim_raw` against the
    plain one; `softmax_p1` is the top-1 share of a softmax over the top-50 scores.
    """
    r = np.arange(len(sim))
    s1 = sim[r, top1]
    k = min(50, sim.shape[1])
    topk = -np.sort(-np.partition(sim, -k, axis=1)[:, -k:], axis=1)
    w = np.exp((topk - s1[:, None]) / 0.05)
    return pd.DataFrame({"cos_sim_raw": sim_raw[r, top1], "softmax_p1": 1.0 / w.sum(1), "s2": topk[:, 1]})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=32, help="crops per forward pass (default: %(default)s)")
    p.add_argument("--num-threads", type=int, default=32, help="decode/preprocess worker threads, -1 for every core (default: %(default)s)")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="confidence a top-1 candidate must beat to be returned (default: %(default)s)")
    args = p.parse_args(argv)

    query_df = pd.read_csv(DATA_DIR / "test_query.csv", dtype={"image_id": str})
    gallery_df = pd.read_csv(DATA_DIR / "test_gallery.csv", dtype={"image_id": str})
    logging.info("%d query, %d gallery from %s", len(query_df), len(gallery_df), DATA_DIR)

    items = [
        {"image_path": str(DATA_DIR / "images" / f"{r.image_id}.jpg"), "bbox": (r.x, r.y, r.w, r.h)}
        for r in pd.concat([query_df, gallery_df]).itertuples()
    ]

    embedder = VehicleEmbedder(checkpoint=CHECKPOINT, num_threads=args.num_threads)

    out = []
    for start in range(0, len(items), args.batch_size):
        out.append(embedder.extract(items[start : start + args.batch_size]))
    emb = np.concatenate(out)
    assert emb.shape == (len(items), 2304), emb.shape

    q_emb = emb[: len(query_df)]
    g_raw = emb[len(query_df) :]  # the confidence model's cos_sim_raw is measured against this
    g_emb = DBA(g_raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "embeddings.npy", np.concatenate([q_emb, g_emb]).astype(np.float32))
    logging.info("saved %s embeddings to %s", emb.shape, OUTPUT_DIR / "embeddings.npy")

    sim = q_emb @ g_emb.T
    top = np.argsort(-sim, axis=1, kind="stable")[:, :TOP_K]
    sub = pd.DataFrame(gallery_df.image_id.to_numpy()[top])
    sub.insert(0, "query_id", query_df.image_id.to_numpy())
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False, header=False)
    logging.info("saved top-%d for %d queries to %s", TOP_K, len(sub), OUTPUT_DIR / "submission.csv")

    model = CatBoostClassifier()
    model.load_model(str(CONFIDENCE_MODEL))
    top1 = top[:, 0]
    FEATURES = ["cos_sim_raw", "softmax_p1", "s2"]
    conf = model.predict_proba(confidence_features(sim, q_emb @ g_raw.T, top1)[FEATURES])[:, 1]

    cand = pd.DataFrame(
        {
            "query_id": query_df.image_id.to_numpy(),
            "gallery_id": gallery_df.image_id.to_numpy()[top1],
            "confidence": conf,
        }
    )
    kept = cand[cand.confidence > args.threshold]
    kept.to_csv(OUTPUT_DIR / "candidates.csv", index=False)
    logging.info(
        "saved %d of %d top-1 candidates above confidence %.3f to %s",
        len(kept),
        len(query_df),
        args.threshold,
        OUTPUT_DIR / "candidates.csv",
    )


if __name__ == "__main__":
    main()
