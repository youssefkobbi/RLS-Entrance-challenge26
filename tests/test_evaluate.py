"""Tests for the tokenizer, the collate function and the attribute parser (src/evaluate.py)."""

import pytest
import torch

from src.data import apply_blind, collate
from src.evaluate import parse_word
from src.tokenizer import EOS_ID, IGNORE_INDEX, CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    ids = tok.encode("largeredcircle")
    assert ids[-1] == EOS_ID and len(ids) == 15
    assert tok.decode(ids) == "largeredcircle"
    assert tok.vocab_size == 27


def test_tokenizer_rejects_non_letters():
    with pytest.raises(ValueError):
        CharTokenizer().encode("large red")


def test_batch_padding_and_targets():
    tok = CharTokenizer()
    inputs, targets = tok.batch(["abc", "a"])
    assert inputs.shape == (2, 3) and targets.shape == (2, 4)
    assert targets[0].tolist() == [0, 1, 2, EOS_ID]
    assert targets[1].tolist() == [0, EOS_ID, IGNORE_INDEX, IGNORE_INDEX]


def test_collate_matches_tokenizer_batch():
    tok = CharTokenizer()
    words = ["largeredcircle", "smallbluesquareleftoflargeyellowtriangle"]
    items = [(torch.zeros(3, 64, 64), torch.tensor(tok.encode(w))) for w in words]
    batch = collate(items)
    inputs, targets = tok.batch(words)
    assert torch.equal(batch["input_ids"], inputs) and torch.equal(batch["targets"], targets)
    assert batch["lengths"].tolist() == [14, 40]


def test_parse_clean_words():
    p = parse_word("largeredcircle")
    assert (p["size1"], p["color1"], p["shape1"], p["n_objects"], p["cost"]) == ("large", "red", "circle", 1, 0)
    p = parse_word("smallbluesquareleftoflargeyellowtriangle")
    assert p["relation"] == "leftof" and p["size2"] == "large" and p["color2"] == "yellow"
    assert p["shape2"] == "triangle" and p["n_objects"] == 2 and p["cost"] == 0


@pytest.mark.parametrize(
    "broken, expected",
    [
        ("largeredcirle", ("large", "red", "circle")),
        ("largeredcircleee", ("large", "red", "circle")),
        ("lrgeredcircle", ("large", "red", "circle")),
        ("largeblue", ("large", "blue", "circle")),  # truncated: shape is a best guess, size/color survive
    ],
)
def test_parse_survives_misspellings(broken, expected):
    p = parse_word(broken)
    assert p["cost"] > 0
    assert (p["size1"], p["color1"]) == expected[:2]
    if broken != "largeblue":
        assert p["shape1"] == expected[2]


def test_parse_misspelled_two_objects():
    p = parse_word("smalbluesquareleftofflargeyelowtriangl")
    assert p["n_objects"] == 2
    assert (p["size1"], p["color1"], p["shape1"], p["relation"]) == ("small", "blue", "square", "leftof")
    assert (p["size2"], p["color2"], p["shape2"]) == ("large", "yellow", "triangle")


def test_parse_empty_word_does_not_crash():
    assert parse_word("")["n_objects"] in (1, 2)


def test_apply_blind():
    x = torch.randn(4, 3, 8, 8)
    assert torch.equal(apply_blind(x, "none"), x)
    assert apply_blind(x, "zeros").abs().sum() == 0
    shuffled = apply_blind(x, "shuffle", torch.Generator().manual_seed(0))
    assert sorted(shuffled.flatten().tolist()) == sorted(x.flatten().tolist())
