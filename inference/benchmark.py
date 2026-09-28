"""Stands in for the organisers' measurement script, and scores the result against their bands.

It touches the solution only through `vehicle_encoder.VehicleEmbedder`: construct once, then call
`extract()` in a loop, the way a scoring harness has to in order to fence each call with a CUDA
sync. Every call is serial — any overlap between decode and forward has to come from inside
`extract`, not from the harness.

    uv run python -m inference.benchmark
    docker run --rm --gpus all --network none --entrypoint python -v /path/to/data:/data:ro \
        madxx31/lct-vehicle-reid:v1 -m inference.benchmark

The knobs live in the constants below rather than in a config file; this is a one-machine script.
The protocol is the published one: latency is the median of 300 batch-1 cycles after 50 warmups,
throughput is the best sustained FPS over batches 1/8/16/32 with at least 10 s on each.
"""

import argparse
import logging
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .vehicle_encoder import DEFAULT_CHECKPOINT, IN_IMAGE, VehicleEmbedder

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

ROOT = Path(__file__).parent.parent
# in the Docker image the data is bind-mounted at /data, as for inference.py
DATA_DIR = Path("/data") if IN_IMAGE else ROOT / "data"
CHECKPOINT = DEFAULT_CHECKPOINT

SPLITS = ("test_query", "test_gallery")
WARMUP, ITERS = 50, 300
MIN_SECONDS = 10.0
BATCH_SIZES = (1, 8, 16, 32)
NUM_THREADS = 32  # decode threads inside extract(), capped at the cores this process may use

# the jury's published scoring bands, in ms and FPS
LATENCY_BAND = (40.0, 80.0)
THROUGHPUT_BAND = (50.0, 100.0)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_items(data_dir):
    """The test vehicles as the harness would hand them over: a full frame plus its BBox."""
    df = pd.concat([pd.read_csv(Path(data_dir) / f"{s}.csv") for s in SPLITS], ignore_index=True)
    img_dir = Path(data_dir) / "images"
    return [{"image_path": str(img_dir / f"{r.image_id}.jpg"), "bbox": [int(r.x), int(r.y), int(r.w), int(r.h)]} for r in df.itertuples()]


def measure_latency(model, items, warmup, iters):
    """Median of `iters` single-image cycles, each fenced by a CUDA sync on both sides."""
    device = model.device
    for i in range(warmup):
        model.extract([items[i % len(items)]])
    sync(device)

    timings = []
    for i in range(iters):
        item = items[(warmup + i) % len(items)]
        sync(device)
        t0 = time.perf_counter()
        model.extract([item])
        sync(device)
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.fmean(timings),
        "p90_ms": timings[int(0.90 * (len(timings) - 1))],
        "p99_ms": timings[int(0.99 * (len(timings) - 1))],
        "min_ms": timings[0],
        "max_ms": timings[-1],
        "n": len(timings),
    }


def measure_throughput(model, items, batch_size, min_seconds, warmup_batches=3):
    """Sustained FPS from back-to-back `extract` calls, nothing overlapping them."""
    device = model.device

    def batch_at(i):
        start = (i * batch_size) % len(items)
        return [items[(start + j) % len(items)] for j in range(batch_size)]

    for i in range(warmup_batches):
        model.extract(batch_at(i))
    sync(device)

    n, i, t0 = 0, warmup_batches, time.perf_counter()
    while time.perf_counter() - t0 < min_seconds:
        model.extract(batch_at(i))
        i, n = i + 1, n + batch_size
    sync(device)
    return n / (time.perf_counter() - t0)


def profile_stages(model, items, iters=100):
    """Where a single-image cycle goes, stage by stage.

    Reaches past `extract` into the stage methods it is built from. Every stage is fenced by a CUDA
    sync, which costs a few tens of microseconds, so the stages sum to slightly more than the
    unfenced latency above.
    """
    device = model.device
    names = ("disk read", "jpeg decode", "bbox crop", "preprocess", "h2d + stack", "forward", "l2 + to cpu")
    stages = {n: [] for n in names}
    for i in range(iters):
        item = items[i % len(items)]
        marks = [time.perf_counter()]

        def mark(value):
            sync(device)
            marks.append(time.perf_counter())
            return value

        data = mark(model.read(item["image_path"]))
        img = mark(model.decode(data))
        img = mark(model.crop(img, item["bbox"]))
        sample = mark(model.preprocess(img))
        batch = mark(model.collate([sample]))
        emb = mark(model.forward(batch))
        mark(emb.cpu().numpy())
        for name, t0, t1 in zip(names, marks, marks[1:]):
            stages[name].append((t1 - t0) * 1000)
    return {
        name: {
            "median_ms": statistics.median(v),
            "mean_ms": statistics.fmean(v),
            "p90_ms": sorted(v)[int(0.9 * (len(v) - 1))],
        }
        for name, v in stages.items()
    }


def print_stage_table(stages):
    total = sum(s["median_ms"] for s in stages.values())
    print(f"{'stage':<14}{'median':>9}{'mean':>9}{'p90':>9}{'share':>8}")
    for name, s in stages.items():
        print(f"{name:<14}{s['median_ms']:>9.2f}{s['mean_ms']:>9.2f}{s['p90_ms']:>9.2f}{s['median_ms'] / total:>8.0%}")
    print(f"{'total':<14}{total:>9.2f}")


def check_determinism(model, items, n=64, batch_size=16):
    """Two passes over identical data: the jury runs this, and bf16 reductions can drift."""

    def run():
        return np.concatenate([model.extract(items[i : i + batch_size]) for i in range(0, n, batch_size)])

    a, b = run(), run()
    return {"bitwise_identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max())}


def linear_score(value, low, high, higher_is_better):
    """The jury's piecewise-linear bands: full marks past the good end, zero past the bad one."""
    if higher_is_better:
        return 1.0 if value >= high else 0.0 if value <= low else (value - low) / (high - low)
    return 1.0 if value <= low else 0.0 if value >= high else (high - value) / (high - low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=NUM_THREADS, help="decode threads inside extract(); -1 takes every core")
    args = parser.parse_args()

    items = load_items(DATA_DIR)
    logging.info(f"{len(items)} test vehicles from {' + '.join(SPLITS)}")

    device = torch.device("cuda")  # VehicleEmbedder refuses to start without it
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model = VehicleEmbedder(checkpoint=CHECKPOINT, device=device, num_threads=args.threads)
    load_seconds = time.perf_counter() - t0
    logging.info(f"loaded {CHECKPOINT} in {load_seconds:.2f}s ({args.threads} decode threads)")

    logging.info(f"latency: {WARMUP} warmup + {ITERS} timed cycles at batch 1")
    latency = measure_latency(model, items, WARMUP, ITERS)
    logging.info(f"  median {latency['median_ms']:.1f} ms (p90 {latency['p90_ms']:.1f}, p99 {latency['p99_ms']:.1f})")

    stages = profile_stages(model, items)  # printed with the summary below

    throughput = {}
    for bs in BATCH_SIZES:
        throughput[bs] = measure_throughput(model, items, bs, MIN_SECONDS)
        logging.info(f"  batch {bs:>2}: {throughput[bs]:.1f} FPS")

    best_fps = max(throughput.values())
    determinism = check_determinism(model, items)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    weights_bytes = CHECKPOINT.stat().st_size

    lat_score = linear_score(latency["median_ms"], *LATENCY_BAND, higher_is_better=False)
    thr_score = linear_score(best_fps, *THROUGHPUT_BAND, higher_is_better=True)
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": {
            "image_size": model.image_size,
            "threads": args.threads,
            "autocast": "bf16" if device.type == "cuda" else "fp32",
        },
        "latency_b1": latency,
        "stage_profile_b1": stages,
        "throughput_fps": throughput,
        "best_fps": best_fps,
        "peak_vram_gb": peak_vram_gb,
        "weights_load_seconds": load_seconds,
        "weights_bytes": weights_bytes,
        "weights_limit_ok": weights_bytes <= 2 * 2**30,
        "determinism": determinism,
        "scores": {
            "latency_score": lat_score,
            "throughput_score": thr_score,
            # performance_score is 20% of the total: 10 points of latency + 10 of throughput
            "performance_points_of_20": 10 * lat_score + 10 * thr_score,
        },
    }

    print()
    print(f"device                {report['device']}")
    print(f"latency  (batch 1)    {latency['median_ms']:.1f} ms median   -> {lat_score:.2f} of 1  (<=40 full, >=80 zero)")
    print(f"throughput (best)     {best_fps:.1f} FPS          -> {thr_score:.2f} of 1  (>=100 full, <=50 zero)")
    print(f"performance_score     {report['scores']['performance_points_of_20']:.1f} / 20 points")
    print(f"peak VRAM             {peak_vram_gb:.2f} GB")
    print(
        f"weights               {weights_bytes / 2**20:.0f} MB, loaded in {load_seconds:.2f}s  (limit 2 GB: {'ok' if report['weights_limit_ok'] else 'FAIL'})"
    )
    verdict = "bit-identical" if determinism["bitwise_identical"] else f"max |diff| {determinism['max_abs_diff']:.2e}"
    print(f"determinism           {verdict}")
    print()
    print("per-stage cost of one batch-1 cycle (ms):")
    print_stage_table(stages)


if __name__ == "__main__":
    main()
