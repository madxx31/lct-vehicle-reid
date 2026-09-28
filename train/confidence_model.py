"""Fit the confidence model on out-of-fold embeddings and pick its threshold.

    python -m train.confidence_model

Reads oof_preds/embeddings34oof.npy (fit, threshold) and oof_preds/embeddings12oof.npy (held-out check),
writes models/confidence_model.cbm and docs/confidence_thresholds.png.
"""

import os
from pathlib import Path

# for reproducibility
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import matplotlib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def DBA(G):
    """Database-side augmentation: blend each gallery vector with its nearest neighbour."""
    s = G @ G.T
    np.fill_diagonal(s, -np.inf)
    i = s.argmax(1)
    sim1 = s[np.arange(len(G)), i]
    return normalize(G + np.where(sim1 > 0.5, sim1.clip(min=0) ** 2, 0)[:, None] * G[i])


def f1_official(score, y_true, has_match, threshold):
    ret = score > threshold
    tp, fp = (ret & (y_true == 1)).sum(), (ret & (y_true == 0)).sum()
    fn = (~ret & has_match).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def best_threshold(score, y, has_match, n=400, tol=1e-4):
    grid = np.quantile(score, np.linspace(0.02, 0.98, n))
    f1 = np.array([f1_official(score, y, has_match, t) for t in grid])
    best = f1.max()
    return float(np.median(grid[f1 >= best - tol])), best


def split_into_query_and_gallery(df, seed, n_vehicles=300, openset_rate=0.2):
    """Returns indexes in df of one simulated gallery/query split.

    gallery: N_VEHICLES vehicles, one image from each camera they appear on
    query:   every remaining image of those vehicles that has a cross-camera positive in the
             gallery, plus images of held-out vehicles filling OPENSET_RATE of the queries
    """
    rng = np.random.default_rng(seed)
    cam = df.camera_id.to_numpy()
    groups = df.groupby("vehicle_id").indices  # vehicle_id -> positions in df
    vehicles = rng.permutation(list(groups))
    gal_v, open_v = vehicles[:n_vehicles], vehicles[n_vehicles:]

    gallery, query = [], []
    for v in gal_v:
        rows = rng.permutation(groups[v])  # random order => random pick per camera
        take = np.array([rows[cam[rows] == c][0] for c in pd.unique(cam[rows])])
        rest = np.setdiff1d(rows, take)
        gallery.append(take)
        query.append(rest[(cam[rest][:, None] != cam[take][None, :]).any(1)])

    n_q = sum(map(len, query))
    n_open = round(openset_rate / (1 - openset_rate) * n_q)
    pool = rng.permutation(np.concatenate([groups[v] for v in open_v]))[:n_open]
    return np.concatenate(gallery), np.concatenate(query + [pool])


def build_sample(q_df, g_df, q_emb, g_emb):
    """For each query get top-1 gallery candidate and features"""
    q_emb, g_emb = normalize(q_emb), normalize(g_emb)
    sim = q_emb @ DBA(g_emb).T
    sim_raw = q_emb @ g_emb.T

    r = np.arange(len(q_emb))
    top = sim.argmax(1)
    s1 = sim[r, top]

    # for feature calculation: similarity with top-2, softmax over top-50
    k = min(50, sim.shape[1])
    topk = -np.sort(-np.partition(sim, -k, axis=1)[:, -k:], axis=1)
    w = np.exp((topk - s1[:, None]) / 0.05)

    return pd.DataFrame(
        {
            "query_id": q_df.image_id.to_numpy(),
            "vehicle_id": q_df.vehicle_id.to_numpy(),
            "gallery_id": g_df.image_id.to_numpy()[top],
            "cos_sim_dba": s1,
            "cos_sim_raw": sim_raw[r, top],
            "softmax_p1": 1.0 / w.sum(1),
            "s2": topk[:, 1],
            "y": (g_df.vehicle_id.to_numpy()[top] == q_df.vehicle_id.to_numpy()).astype(int),
            "has_match": (
                (q_df.vehicle_id.to_numpy()[:, None] == g_df.vehicle_id.to_numpy()[None, :])
                & (q_df.camera_id.to_numpy()[:, None] != g_df.camera_id.to_numpy()[None, :])
            ).any(1),
        }
    )


def build_samples(folds, emb_path, trials=100):
    df = pd.read_csv(ROOT / "data" / "train.csv")
    df = df[(df.vehicle_id % 5).isin(folds)].reset_index(drop=True)
    emb = np.load(emb_path)
    assert len(emb) == len(df), (len(emb), len(df))

    out = []
    for trial in range(trials):
        g_idx, q_idx = split_into_query_and_gallery(df, seed=trial)
        sample = build_sample(df.iloc[q_idx], df.iloc[g_idx], emb[q_idx], emb[g_idx])
        sample["trial"] = trial
        out.append(sample)
    return pd.concat(out, ignore_index=True)


def plot_thresholds(s, thresholds, path, n=200):
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
    HUE = {"conf": "#2a78d6", "cos_sim_dba": "#eb6834"}
    LABEL = {"conf": "F1 при отсечении\nпо модели уверенности", "cos_sim_dba": "F1 при отсечении\nпо косинусной близости"}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y, has_match = s.y.to_numpy(), s.has_match.to_numpy()
    peaks = []
    for col, t in thresholds.items():
        score = s[col].to_numpy()
        grid = np.quantile(score, np.linspace(0.02, 0.995, n))
        f1 = [f1_official(score, y, has_match, x) for x in grid]
        peaks.append(max(f1))
        ax.plot(grid, f1, color=HUE[col], lw=2, label=LABEL[col])

        picked = f1_official(score, y, has_match, t)
        ax.axvline(t, color=HUE[col], lw=1.1, ls="--")
        ax.plot([t], [picked], "o", color=HUE[col], ms=6)
        ax.annotate(
            f"порог {t:.3f}\nF1 {picked:.4f}",
            (t, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -6),
            textcoords="offset points",
            va="top",
            color=HUE[col],
            fontsize=9,
        )
    # the tails plunge to 0 and would flatten the only part worth reading
    ax.set_ylim(min(peaks) - 0.06, max(peaks) + 0.012)
    ax.set_xlabel("Порог", color=MUTED)
    ax.set_ylabel("F1", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(fontsize=9, frameon=False, labelcolor=MUTED, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("F1 в зависимости от порога на косинусную близость и на модель уверенности", color=INK, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nplot -> {path}")


# train on folds 3,4, validate on folds 1,2
trn = build_samples((3, 4), ROOT / "oof_preds" / "embeddings34oof.npy")
val = build_samples((1, 2), ROOT / "oof_preds" / "embeddings12oof.npy")

FEATURES = ["cos_sim_raw", "softmax_p1", "s2"]
model = CatBoostClassifier(learning_rate=0.1, num_trees=60, verbose=False, random_seed=0, allow_writing_files=False, thread_count=1)
model.fit(trn[FEATURES], trn.y)
for s in (trn, val):
    s["conf"] = model.predict_proba(s[FEATURES])[:, 1]

y_trn, has_match_trn = trn.y.to_numpy(), trn.has_match.to_numpy()
t_dba, _ = best_threshold(trn.cos_sim_dba.to_numpy(), y_trn, has_match_trn)
t_conf, _ = best_threshold(trn.conf.to_numpy(), y_trn, has_match_trn)
print("thresholds picked on folds 3-4 to maximize F1:")
print(f"  cos_sim_dba > {t_dba:.4f}")
print(f"  conf        > {t_conf:.4f}")

for name, s in [("folds 3-4 (fitted)", trn), ("folds 1-2 (held out)", val)]:
    y, has_match = s.y.to_numpy(), s.has_match.to_numpy()
    dba = f1_official(s.cos_sim_dba.to_numpy(), y, has_match, t_dba)
    conf = f1_official(s.conf.to_numpy(), y, has_match, t_conf)
    print(f"F1 on {name}: cos_sim_dba {dba:.4f} -> conf {conf:.4f} ({conf - dba:+.4f})")

plot_thresholds(val, {"cos_sim_dba": t_dba, "conf": t_conf}, ROOT / "docs" / "confidence_thresholds.png")

model_path = ROOT / "models" / "confidence_model.cbm"
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(model_path))
print(f"model -> {model_path}")
