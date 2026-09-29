"""S1 -- training throughput (images / second) versus batch size, on CPU and GPU.

Usage (from the repository root):
    python -m benchmarks.S1_throughput.benchmark                       # every available device
    python -m benchmarks.S1_throughput.benchmark --devices cpu --batch-sizes 1 16 64 256

What is timed: one full TRAINING step = forward + loss + backward + optimizer step, on a batch that is already in
device memory ("compute" mode).  Data loading is deliberately OUTSIDE the timer, so the number isolates the model.
With `--with-loader` the DataLoader (indexing, collate) and the CPU->GPU copy are timed too ("loader" mode).

The three measurement traps (Project Brief, section 9):
1. Warm-up: the first iterations pay one-off costs (CUDA context / cuDNN autotune / memory-allocator growth /
   lazy library loading / CPU caches).  `--warmup` steps are run and thrown away.
2. Asynchronous execution: CUDA calls return before the GPU finishes.  torch.cuda.synchronize() is called
   before the timer starts and before it is read.  (A no-op on CPU.)
3. Data loading: outside the timer by default (see above), inside with --with-loader.

Each (device, batch size) is repeated `--repeats` times; the raw per-repeat timings go to results.csv, and the
plot shows mean +- std.  hardware.txt records the exact CPU / GPU / library versions.
"""

import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from src.data import ShapeScenes, collate
from src.tokenizer import IGNORE_INDEX, VOCAB_SIZE
from src.train import compute_loss
from src.utils import build_model, load_config, set_seed

HERE = Path(__file__).resolve().parent


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()  # wait until every queued GPU kernel has finished


def train_step(model, optimizer, images, input_ids, targets):
    logits = model(images, input_ids)
    loss = compute_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def bench_compute(model, optimizer, batch, device, warmup, iters):
    """Seconds for `iters` training steps on one fixed batch that already lives on `device`."""
    images, input_ids, targets = (batch[k].to(device) for k in ("images", "input_ids", "targets"))
    for _ in range(warmup):
        train_step(model, optimizer, images, input_ids, targets)
    sync(device)  # make sure warm-up work is finished before the timer starts
    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(model, optimizer, images, input_ids, targets)
    sync(device)  # ... and that the timed work is finished before the timer is read
    return time.perf_counter() - t0


def bench_loader(model, optimizer, dataset, batch_size, device, warmup, iters):
    """Same, but the batch is built from the dataset (indexing + collate) and copied to the device inside the timer."""
    n = len(dataset)

    def one_step(i):
        idx = [(i * batch_size + j) % n for j in range(batch_size)]
        batch = collate([dataset[k] for k in idx])
        images, input_ids, targets = (batch[k].to(device) for k in ("images", "input_ids", "targets"))
        train_step(model, optimizer, images, input_ids, targets)

    for i in range(warmup):
        one_step(i)
    sync(device)
    t0 = time.perf_counter()
    for i in range(iters):
        one_step(warmup + i)
    sync(device)
    return time.perf_counter() - t0


def hardware_report(devices):
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", f"platform: {platform.platform()}",
             f"python: {platform.python_version()}", f"torch: {torch.__version__}",
             f"torch cpu threads: {torch.get_num_threads()}"]
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        cpu = platform.processor() or "unknown"
    lines.append(f"cpu: {cpu}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines += [f"gpu: {p.name} ({p.total_memory / 2**30:.1f} GiB)", f"cuda: {torch.version.cuda}",
                  f"cudnn: {torch.backends.cudnn.version()}"]
    for cmd in (["lscpu"], ["nvidia-smi"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
            lines += ["", f"$ {' '.join(cmd)}", out.strip()]
        except Exception:
            pass
    return "\n".join(lines)


def make_plot(rows, path, mode):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for device in sorted({r["device"] for r in rows}):
        sizes = sorted({r["batch_size"] for r in rows if r["device"] == device})
        means, stds = [], []
        for b in sizes:
            vals = [r["images_per_s"] for r in rows if r["device"] == device and r["batch_size"] == b]
            means.append(statistics.mean(vals))
            stds.append(statistics.pstdev(vals))
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=device)
    ax.set(xscale="log", yscale="log", xlabel="batch size", ylabel="training throughput (images / s)",
           title=f"S1: training throughput ({mode} timing, mean +- std over repeats)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--devices", nargs="+", default=None, help="cpu cuda (default: all available)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--warmup", type=int, default=5, help="discarded steps before timing")
    parser.add_argument("--iters", type=int, default=10, help="timed steps per repeat")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--with-loader", action="store_true", help="include DataLoader work + host->device copy")
    parser.add_argument("--out-dir", default=str(HERE))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    mode = "loader" if args.with_loader else "compute"
    cfg = load_config(args.config)
    dataset = ShapeScenes("train", args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hardware.txt").write_text(hardware_report(devices) + "\n")

    rows = []
    for dev_name in devices:
        device = torch.device(dev_name)
        for bs in args.batch_sizes:
            set_seed(0)
            model = build_model(cfg).to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            batch = collate([dataset[i] for i in range(bs)])
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for rep in range(args.repeats):
                if args.with_loader:
                    secs = bench_loader(model, optimizer, dataset, bs, device, args.warmup if rep == 0 else 1,
                                        args.iters)
                else:
                    secs = bench_compute(model, optimizer, batch, device, args.warmup if rep == 0 else 1, args.iters)
                row = {"device": dev_name, "mode": mode, "batch_size": bs, "repeat": rep, "iters": args.iters,
                       "seconds": secs, "ms_per_step": 1000 * secs / args.iters,
                       "images_per_s": bs * args.iters / secs,
                       "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else ""}
                rows.append(row)
                print(f"{dev_name:5s} bs={bs:4d} rep={rep}  {row['ms_per_step']:9.2f} ms/step  "
                      f"{row['images_per_s']:10.1f} img/s", flush=True)
            del model, optimizer

    with open(out_dir / f"results_{mode}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plot(rows, out_dir / f"throughput_{mode}.png", mode)
    print("wrote", out_dir / f"results_{mode}.csv", "and", out_dir / f"throughput_{mode}.png")


if __name__ == "__main__":
    main()
