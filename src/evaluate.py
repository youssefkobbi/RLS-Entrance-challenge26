"""Evaluation: exact-match and per-attribute accuracy.

Includes the parser that splits a (possibly misspelled) generated word
back into size / color / shape [/ relation / size / color / shape].

Usage:
    python -m src.evaluate --checkpoint runs/baseline_seed0/best.pt --split test
    python -m src.evaluate --checkpoint runs/baseline_seed0/best.pt --split test test_heldout

Metrics (see the Project Brief, section 5)
* exact_match      generated word == reference word.
* attributes       parse the generated word into (size, color, shape [, relation, size, color, shape]) and report
                   accuracy per attribute type.  size / color / shape are pooled over both objects (object 1 and
                   object 2 are also reported separately in `per_field`); relation is over the 2-object references.
* letter accuracy  TEACHER-FORCED: the model is fed the TRUE previous letters and we check its arg-max for the next
                   one.  It is split into "first letter of a part" (needs the image: is it l(arge) or s(mall)?) and
                   "inside a part" (`ircle` after `c` is pure spelling).  The inside-part accuracy is high even for
                   a model that never looks at the image, which is why plain letter accuracy flatters the model.
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import torch

from src.data import ShapeScenes, apply_blind, collate
from src.generate import greedy_generate, load_checkpoint
from src.tokenizer import IGNORE_INDEX, CharTokenizer
from src.utils import get_device

SIZES = ("small", "large")
COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle", "cross")
RELATIONS = ("leftof", "rightof", "above", "below")

OBJECT_SLOTS = (SIZES, COLORS, SHAPES)  # lexicon of each slot of one object
SLOTS_ONE = OBJECT_SLOTS
SLOTS_TWO = OBJECT_SLOTS + (RELATIONS,) + OBJECT_SLOTS
FIELDS_ONE = ("size1", "color1", "shape1")
FIELDS_TWO = FIELDS_ONE + ("relation", "size2", "color2", "shape2")
FIELDS = FIELDS_TWO
MAX_PART_LEN = max(len(w) for lex in SLOTS_TWO for w in lex) + 3  # a part can be up to 3 letters too long

_EXACT = {
    1: re.compile("(%s)(%s)(%s)" % tuple("|".join(lex) for lex in SLOTS_ONE)),
    2: re.compile("(%s)(%s)(%s)(%s)(%s)(%s)(%s)" % tuple("|".join(lex) for lex in SLOTS_TWO)),
}


@lru_cache(maxsize=None)
def _edit_distance(a, b):
    """Levenshtein distance (insertions, deletions, substitutions all cost 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=None)
def _nearest(lexicon, text):
    """(distance, word) of the lexicon entry closest to `text` (ties -> first entry)."""
    return min((_edit_distance(text, w), w) for w in lexicon)


def _segment(word, slots):
    """Cut `word` into len(slots) consecutive parts, minimising the total edit distance of each part to the
    nearest word of its slot's lexicon (dynamic programming over the cut positions).

    Returns (total_cost, [best lexicon word for each slot]).
    """
    n, k = len(word), len(slots)
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(k + 1)]  # cost[s][i]: best cost of parsing word[:i] into s slots
    back = [[0] * (n + 1) for _ in range(k + 1)]  # back[s][i]: where slot s-1 started
    cost[0][0] = 0
    for s in range(1, k + 1):
        for i in range(n + 1):
            for j in range(max(0, i - MAX_PART_LEN), i + 1):
                if cost[s - 1][j] == inf:
                    continue
                c = cost[s - 1][j] + _nearest(slots[s - 1], word[j:i])[0]
                if c < cost[s][i]:
                    cost[s][i], back[s][i] = c, j
    cuts, i = [], n
    for s in range(k, 0, -1):
        j = back[s][i]
        cuts.append((j, i))
        i = j
    cuts.reverse()
    return cost[k][n], [_nearest(slots[s], word[a:b])[1] for s, (a, b) in enumerate(cuts)]


def parse_word(word):
    """Parse a (possibly broken) word into its attributes.

    Returns a dict with keys size1, color1, shape1, relation, size2, color2, shape2 (None when absent), plus
    `n_objects` (1 or 2) and `cost` (0 for a perfectly spelled word, else the total edit distance to the nearest
    valid parse).  Works on outputs such as 'largeredcirle', 'smalbluesquareleftoflargeyelowtriangle' or ''.
    """
    for n_obj, pattern in _EXACT.items():  # fast path: a well-formed word
        m = pattern.fullmatch(word)
        if m:
            return _pack(list(m.groups()), n_obj, 0)
    cost1, words1 = _segment(word, SLOTS_ONE)
    cost2, words2 = _segment(word, SLOTS_TWO)
    if cost1 <= cost2:  # tie -> fewer objects
        return _pack(words1, 1, cost1)
    return _pack(words2, 2, cost2)


def _pack(values, n_objects, cost):
    fields = FIELDS_ONE if n_objects == 1 else FIELDS_TWO
    out = {f: None for f in FIELDS}
    out.update(dict(zip(fields, values)))
    out["n_objects"] = n_objects
    out["cost"] = cost
    return out


def _part_starts(reference):
    """Indices (in the reference word) at which a new part (size / color / shape / relation ...) starts."""
    parsed = parse_word(reference)
    fields = FIELDS_ONE if parsed["n_objects"] == 1 else FIELDS_TWO
    starts, pos = set(), 0
    for f in fields:
        starts.add(pos)
        pos += len(parsed[f])
    return starts


@torch.no_grad()
def evaluate_model(model, dataset, device, batch_size=250, blind="none", max_examples=30, seed=0):
    """Full evaluation of a model on one dataset.  Returns a JSON-serialisable dict."""
    tokenizer = CharTokenizer()
    model.eval()
    gen = torch.Generator().manual_seed(seed)  # for the 'shuffle' blind mode

    n = len(dataset)
    exact = 0
    exact_by_obj = {1: [0, 0], 2: [0, 0]}  # [correct, total]
    field_hits = {f: [0, 0] for f in FIELDS}
    pooled = {"size": [0, 0], "color": [0, 0], "shape": [0, 0], "relation": [0, 0], "n_objects": [0, 0]}
    malformed = 0
    errors = []
    tf_hits = {"first": [0, 0], "inside": [0, 0]}  # teacher-forced letter accuracy

    for start in range(0, n, batch_size):
        batch = collate([dataset[i] for i in range(start, min(n, start + batch_size))])
        images = apply_blind(batch["images"].to(device), blind, gen)
        words_ref = dataset.words[start : start + len(images)]

        # ---- free-running greedy generation -> exact match + attributes
        words_gen, _ = greedy_generate(model, images, tokenizer)
        for ref, hyp in zip(words_ref, words_gen):
            pr, ph = parse_word(ref), parse_word(hyp)
            ok = hyp == ref
            exact += ok
            exact_by_obj[pr["n_objects"]][0] += ok
            exact_by_obj[pr["n_objects"]][1] += 1
            malformed += ph["cost"] > 0
            for f in FIELDS:
                if pr[f] is None:
                    continue
                hit = ph[f] == pr[f]
                field_hits[f][0] += hit
                field_hits[f][1] += 1
                kind = "relation" if f == "relation" else f[:-1]
                pooled[kind][0] += hit
                pooled[kind][1] += 1
            pooled["n_objects"][0] += ph["n_objects"] == pr["n_objects"]
            pooled["n_objects"][1] += 1
            if not ok and len(errors) < max_examples:
                errors.append({"reference": ref, "generated": hyp, "parsed_generated": {f: ph[f] for f in FIELDS}})

        # ---- teacher-forced letter accuracy (true prefix fed to the model)
        logits = model(images, batch["input_ids"].to(device))  # (B, L_max + 1, 27)
        targets = batch["targets"].to(device)  # (B, L_max + 1)
        pred = logits.argmax(dim=-1)
        for row, ref in enumerate(words_ref):
            starts = _part_starts(ref)
            for pos in range(len(ref)):  # letters only, the <eos> position is skipped
                assert targets[row, pos] != IGNORE_INDEX
                key = "first" if pos in starts else "inside"
                tf_hits[key][0] += int(pred[row, pos] == targets[row, pos])
                tf_hits[key][1] += 1

    def ratio(pair):
        return pair[0] / pair[1] if pair[1] else None

    all_letters = [tf_hits["first"][0] + tf_hits["inside"][0], tf_hits["first"][1] + tf_hits["inside"][1]]
    return {
        "n": n,
        "blind": blind,
        "exact_match": exact / n,
        "exact_match_by_n_objects": {str(k): ratio(v) for k, v in exact_by_obj.items()},
        "attribute_acc": {k: ratio(v) for k, v in pooled.items()},
        "per_field_acc": {f: ratio(v) for f, v in field_hits.items()},
        "letter_acc_teacher_forced": ratio(all_letters),
        "letter_acc_first_of_part": ratio(tf_hits["first"]),
        "letter_acc_inside_part": ratio(tf_hits["inside"]),
        "malformed_output_rate": malformed / n,
        "error_examples": errors,
    }


def format_report(name, res):
    a = res["attribute_acc"]
    lines = [
        f"== {name}  (n={res['n']}, blind={res['blind']})",
        f"exact match          {res['exact_match']:.4f}   (1 obj: {res['exact_match_by_n_objects']['1']:.4f}, "
        f"2 obj: {res['exact_match_by_n_objects']['2']:.4f})",
        "attributes           " + "  ".join(f"{k}={v:.4f}" for k, v in a.items() if v is not None),
        f"letter acc (TF)      all={res['letter_acc_teacher_forced']:.4f}  first-of-part="
        f"{res['letter_acc_first_of_part']:.4f}  inside-part={res['letter_acc_inside_part']:.4f}",
        f"malformed outputs    {res['malformed_output_rate']:.4f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", nargs="+", default=["test"], choices=["val", "test", "test_heldout"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--out-dir", default=None, help="where to write results_<split>.json (default: checkpoint dir)")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    blind = cfg.get("blind", "none")
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent)
    for split in args.split:
        res = evaluate_model(model, ShapeScenes(split, args.data_dir), device, args.batch_size, blind)
        res["split"], res["checkpoint"] = split, str(args.checkpoint)
        print(format_report(split, res))
        with open(out_dir / f"results_{split}.json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
