"""Training loop.

Forward pass, cross-entropy loss with padding excluded via ignore_index,
backward pass, optimizer step, periodic validation, and checkpointing.

Usage (run from the repository root):
    python -m src.train --config configs/baseline.yaml --seed 0          # main model
    python -m src.train --config configs/blind.yaml --seed 0             # E1 blind baseline
    python -m src.train --config configs/baseline.yaml --overfit-one-batch   # E0 sanity check
    python -m src.train --config configs/baseline.yaml --seed 0 --resume     # continue after a Colab crash

Outputs go to runs/<config name>_seed<seed>/ : last.pt, best.pt, history.json, loss_curves.png,
config.yaml, results_val.json, results_test.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.data import ShapeScenes, apply_blind, collate, make_loader
from src.evaluate import evaluate_model, format_report
from src.generate import greedy_generate, load_checkpoint
from src.tokenizer import IGNORE_INDEX, VOCAB_SIZE
from src.utils import build_model, count_parameters, get_device, load_config, set_seed


def compute_loss(logits, targets):
    """logits (B, L + 1, 27), targets (B, L + 1) with IGNORE_INDEX on padding -> scalar mean loss over real tokens."""
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=IGNORE_INDEX)


def make_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=0.05):
    """Linear warm-up, then cosine decay down to min_lr_ratio * lr."""

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def validation_loss(model, loader, device, blind, amp):
    model.eval()
    total, count = 0.0, 0
    gen = torch.Generator().manual_seed(0)
    for batch in loader:
        images = apply_blind(batch["images"].to(device), blind, gen)
        targets = batch["targets"].to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images, batch["input_ids"].to(device))
        n_tokens = (targets != IGNORE_INDEX).sum().item()
        total += compute_loss(logits.float(), targets).item() * n_tokens  # token-weighted average
        count += n_tokens
    return total / count


@torch.no_grad()
def quick_exact_match(model, dataset, device, blind, n=500):
    """Greedy exact-match on the first n samples (cheap progress signal during training)."""
    model.eval()
    batch = collate([dataset[i] for i in range(min(n, len(dataset)))])
    images = apply_blind(batch["images"].to(device), blind, torch.Generator().manual_seed(0))
    words, _ = greedy_generate(model, images)
    return sum(w == r for w, r in zip(words, dataset.words[: len(words)])) / len(words)


def plot_curves(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["step"], history["train_loss_step"], alpha=0.35, label="train (per logged step)")
    ax[0].plot(history["epoch_end_step"], history["train_loss"], "o-", label="train (epoch mean)")
    ax[0].plot(history["epoch_end_step"], history["val_loss"], "s-", label="validation")
    ax[0].set(xlabel="optimizer step", ylabel="cross-entropy (nats / letter)", title="Loss", yscale="log")
    ax[0].legend()
    if history["val_exact_epoch"]:
        ax[1].plot(history["val_exact_epoch"], history["val_exact"], "o-")
    ax[1].set(xlabel="epoch", ylabel="exact match (val, first 500)", title="Validation exact match", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_overfit_one_batch(cfg, device, args):
    """E0: can the pipeline memorise ONE fixed batch?  If not, there is a bug (mask, targets, frozen layer ...)."""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["model"]["dropout"] = 0.0  # regularisation would only get in the way of memorisation
    set_seed(cfg["seed"])
    out_dir = Path(args.out_dir or "experiments/E0_overfit")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ShapeScenes("train", cfg["data"]["dir"])
    batch = collate([dataset[i] for i in range(args.overfit_batch_size)])
    images, input_ids, targets = (batch[k].to(device) for k in ("images", "input_ids", "targets"))

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.overfit_lr, weight_decay=0.0)
    print(f"E0: {count_parameters(model):,} parameters, batch {tuple(images.shape)}, device {device}")

    losses = []
    model.train()
    for step in range(args.overfit_steps):
        logits = model(images, input_ids)  # (B, L + 1, 27)
        loss = compute_loss(logits, targets)
        optimizer.zero_grad(set_to_none=True)  # 1) forget the previous gradients
        loss.backward()  # 2) fill .grad of every parameter (chain rule through the graph)
        optimizer.step()  # 3) move each parameter against its gradient
        losses.append(loss.item())
        if step % 25 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.5f}")

    words, _ = greedy_generate(model, images)
    exact = sum(w == r for w, r in zip(words, dataset.words[: len(words)])) / len(words)
    print(f"E0 result: final loss {losses[-1]:.5f}, greedy exact match on the memorised batch {exact:.3f}")
    for w, r in list(zip(words, dataset.words))[:4]:
        print(f"   generated {w!r}  reference {r!r}")

    with open(out_dir / "e0_log.json", "w") as f:
        json.dump({"losses": losses, "final_exact_match": exact, "batch_size": args.overfit_batch_size,
                   "lr": args.overfit_lr, "seed": cfg["seed"], "device": str(device)}, f)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set(xlabel="optimizer step", ylabel="cross-entropy (nats / letter)", yscale="log",
           title=f"E0: overfit one batch of {args.overfit_batch_size}")
    ax.axhline(math.log(VOCAB_SIZE), ls="--", c="gray", label="ln(27): uniform guessing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "e0_overfit.png", dpi=140)
    print("saved", out_dir / "e0_overfit.png")


def train(cfg, device, args):
    tcfg = cfg["train"]
    blind = cfg.get("blind", "none")
    seed = cfg["seed"]
    name = Path(args.config).stem
    out_dir = Path(args.out_dir or f"runs/{name}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    set_seed(seed)
    data_dir = cfg["data"]["dir"]
    workers = cfg["data"].get("num_workers", 0)
    train_loader = make_loader("train", tcfg["batch_size"], shuffle=True, data_dir=data_dir, num_workers=workers,
                               seed=seed, pin_memory=device.type == "cuda")
    val_loader = make_loader("val", 250, data_dir=data_dir)
    val_set = val_loader.dataset

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, tcfg["warmup_steps"])
    amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"{name} seed {seed} | blind={blind} | {count_parameters(model):,} parameters | device {device} | "
          f"{len(train_loader)} steps/epoch x {tcfg['epochs']} epochs")

    history = {"step": [], "train_loss_step": [], "epoch_end_step": [], "train_loss": [], "val_loss": [],
               "val_exact_epoch": [], "val_exact": [], "epoch_seconds": []}
    start_epoch, best_val, step = 0, float("inf"), 0
    if args.resume and (out_dir / "last.pt").exists():
        ckpt = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history, start_epoch, best_val, step = ckpt["history"], ckpt["epoch"] + 1, ckpt["best_val"], ckpt["step"]
        print(f"resumed from epoch {start_epoch}")

    blind_gen = torch.Generator().manual_seed(seed)
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        t0, running, n_batches = time.time(), 0.0, 0
        for batch in train_loader:
            images = apply_blind(batch["images"].to(device, non_blocking=True), blind, blind_gen)  # (B, 3, 64, 64)
            input_ids = batch["input_ids"].to(device, non_blocking=True)  # (B, L_max)
            targets = batch["targets"].to(device, non_blocking=True)  # (B, L_max + 1)

            with torch.autocast(device.type, enabled=amp):
                logits = model(images, input_ids)  # (B, L_max + 1, 27)
            loss = compute_loss(logits.float(), targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])  # avoid rare exploding updates
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            running += loss.item()
            n_batches += 1
            if step % tcfg.get("log_every", 25) == 0:
                history["step"].append(step)
                history["train_loss_step"].append(loss.item())

        val_loss = validation_loss(model, val_loader, device, blind, amp)
        history["epoch_end_step"].append(step)
        history["train_loss"].append(running / n_batches)
        history["val_loss"].append(val_loss)
        history["epoch_seconds"].append(time.time() - t0)
        msg = f"epoch {epoch + 1:3d}/{tcfg['epochs']}  train {running / n_batches:.4f}  val {val_loss:.4f}  " \
              f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.1f}s"
        if (epoch + 1) % tcfg.get("gen_eval_every", 5) == 0 or epoch + 1 == tcfg["epochs"]:
            em = quick_exact_match(model, val_set, device, blind)
            history["val_exact_epoch"].append(epoch + 1)
            history["val_exact"].append(em)
            msg += f"  val exact(500) {em:.3f}"
        print(msg, flush=True)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "epoch": epoch, "step": step, "history": history, "config": cfg, "best_val": min(best_val, val_loss)}
        torch.save(state, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, out_dir / "best.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f)
        plot_curves(history, out_dir / "loss_curves.png")

    if args.skip_final_eval:
        return
    model, _ = load_checkpoint(out_dir / "best.pt", device)  # evaluate the best-validation checkpoint
    for split in ("val", "test"):
        res = evaluate_model(model, ShapeScenes(split, data_dir), device, blind=blind)
        res.update(split=split, seed=seed, config=str(args.config))
        print(format_report(split, res))
        with open(out_dir / f"results_{split}.json", "w") as f:
            json.dump(res, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides `seed` in the config")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--epochs", type=int, default=None, help="overrides train.epochs (handy for smoke tests)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--overfit-one-batch", action="store_true", help="E0 sanity check instead of real training")
    parser.add_argument("--overfit-steps", type=int, default=400)
    parser.add_argument("--overfit-batch-size", type=int, default=32)
    parser.add_argument("--overfit-lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    device = get_device(args.device)
    if args.overfit_one_batch:
        run_overfit_one_batch(cfg, device, args)
    else:
        train(cfg, device, args)


if __name__ == "__main__":
    main()
