"""PyTorch Dataset and collate function for ShapeScenes.

Loads the image + word pairs written by generate_data.py and returns
(image tensor, token id tensor) pairs, batched with a collate function
that pads to the longest word in the batch.

Run `python -m src.data` to print one decoded sample (checkpoint requirement).
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import CharTokenizer

SPLITS = ("train", "val", "test", "test_heldout")


class ShapeScenes(Dataset):
    """One split of ShapeScenes held entirely in memory as a uint8 tensor (N, 3, 64, 64)."""

    def __init__(self, split, data_dir="data", tokenizer=None):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
        blob = torch.load(Path(data_dir) / f"{split}.pt", weights_only=True)
        self.images = blob["images"]  # (N, 3, 64, 64) uint8, kept as uint8 to save memory
        self.words = blob["words"]  # list[str], length N
        self.tokenizer = tokenizer or CharTokenizer()
        assert len(self.images) == len(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # uint8 in [0, 255] -> float32 in [-1, 1]
        image = self.images[idx].float() / 127.5 - 1.0  # (3, 64, 64)
        ids = torch.tensor(self.tokenizer.encode(self.words[idx], add_eos=True), dtype=torch.long)  # (L + 1,)
        return image, ids


def collate(batch, pad_id=0, ignore_index=-100):
    """Stack images and pad the word ids of a list of (image, ids) pairs.

    Returns a dict:
        images:    (B, 3, 64, 64) float32
        input_ids: (B, L_max)     letters (no <eos>), padded with pad_id
        targets:   (B, L_max + 1) letters + <eos>, padded with ignore_index
        lengths:   (B,)           number of letters in each word
    """
    images = torch.stack([img for img, _ in batch])  # (B, 3, 64, 64)
    lengths = torch.tensor([len(ids) - 1 for _, ids in batch])  # ids include <eos>
    l_max = int(lengths.max())
    input_ids = torch.full((len(batch), l_max), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), l_max + 1), ignore_index, dtype=torch.long)
    for row, (_, ids) in enumerate(batch):
        n = len(ids) - 1
        input_ids[row, :n] = ids[:-1]  # drop <eos> from the input
        targets[row, : n + 1] = ids  # keep <eos> in the target
    return {"images": images, "input_ids": input_ids, "targets": targets, "lengths": lengths}


def make_loader(split, batch_size, shuffle=False, data_dir="data", num_workers=0, seed=0, pin_memory=False):
    dataset = ShapeScenes(split, data_dir)
    generator = torch.Generator().manual_seed(seed)  # makes the shuffling order reproducible
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )


def apply_blind(images, mode, generator=None):
    """E1 blind baseline: destroy the image information but keep the pipeline identical.

    mode "none"    -> images unchanged
    mode "zeros"   -> every image replaced by zeros
    mode "shuffle" -> images permuted inside the batch, so each word is paired with a random image
    """
    if mode in (None, "none"):
        return images
    if mode == "zeros":
        return torch.zeros_like(images)
    if mode == "shuffle":
        perm = torch.randperm(len(images), generator=generator).to(images.device)
        return images[perm]
    raise ValueError(f"unknown blind mode {mode!r}")


if __name__ == "__main__":
    ds = ShapeScenes("train")
    tok = ds.tokenizer
    image, ids = ds[0]
    print("image:", tuple(image.shape), image.dtype, f"range [{image.min():.2f}, {image.max():.2f}]")
    print("word :", ds.words[0])
    print("ids  :", ids.tolist())
    print("back :", tok.decode(ids.tolist()))
    batch = collate([ds[i] for i in range(4)])
    for name, value in batch.items():
        print(f"{name:10s}", tuple(value.shape), value.dtype)
    print("input_ids[0]:", batch["input_ids"][0].tolist())
    print("targets[0]  :", batch["targets"][0].tolist())
