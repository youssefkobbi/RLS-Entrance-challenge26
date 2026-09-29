"""Greedy decoding.

Generates one letter at a time from the visual tokens until <eos> or
45 letters, for inference and for evaluation.

Usage (print a few test-set predictions from a trained checkpoint):
    python -m src.generate --checkpoint runs/baseline_seed0/best.pt --split test -n 8
"""

import argparse

import torch

from src.data import ShapeScenes, apply_blind
from src.tokenizer import MAX_LETTERS, CharTokenizer
from src.utils import build_model, get_device


@torch.no_grad()
def greedy_generate(model, images, tokenizer=None, max_letters=MAX_LETTERS):
    """Greedy decoding for a batch of images.

    images: (B, 3, 64, 64) already on the model's device.
    Returns (words: list[str], ids: (B, n_steps) long tensor of generated ids, <eos> included if produced).

    Each step feeds [visual tokens] + [letters so far] through the decoder and takes the arg-max of the
    LAST logit.  The CNN encoder runs once; only the decoder is re-run at each step (no KV cache: simple > fast).
    """
    tokenizer = tokenizer or CharTokenizer()
    was_training = model.training
    model.eval()
    visual = model.encode_image(images)  # (B, N, d_model), computed once
    batch = images.shape[0]
    ids = torch.empty(batch, 0, dtype=torch.long, device=images.device)  # (B, 0): nothing generated yet
    finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
    for _ in range(max_letters):
        logits = model.decode(visual, ids)  # (B, t + 1, 27)
        next_id = logits[:, -1].argmax(dim=-1)  # (B,) most likely next token
        next_id = torch.where(finished, torch.full_like(next_id, tokenizer.eos_id), next_id)  # freeze finished rows
        ids = torch.cat([ids, next_id[:, None]], dim=1)  # (B, t + 1)
        finished |= next_id == tokenizer.eos_id
        if bool(finished.all()):
            break
    model.train(was_training)
    return [tokenizer.decode(row.tolist()) for row in ids], ids


def load_checkpoint(path, device):
    """Rebuild the model from the config stored in a checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own file
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    ds = ShapeScenes(args.split, args.data_dir)
    images = torch.stack([ds[i][0] for i in range(args.n)]).to(device)
    images = apply_blind(images, cfg.get("blind", "none"))
    words, _ = greedy_generate(model, images)
    for i, word in enumerate(words):
        mark = "OK " if word == ds.words[i] else "BAD"
        print(f"[{mark}] predicted={word!r:48} reference={ds.words[i]!r}")


if __name__ == "__main__":
    main()
