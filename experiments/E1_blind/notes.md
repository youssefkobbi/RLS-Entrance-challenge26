# E1 — Blind baseline

## Question
Does the main model actually use the image, or could a decoder that only knows the spelling statistics of the
words do as well?

## Hypothesis and prediction
I expect the blind model (images replaced by zeros, same architecture, same training) to learn the *language* of
the dataset but not the *scene*, because with zero images the only information left is the letters generated so
far, and the word grammar is size + color + shape [+ relation + size + color + shape].

Predictions for the test split:

* `size` accuracy ~ 0.50, `color` ~ 0.25, `shape` ~ 0.25, `relation` ~ 0.25, `#objects` ~ 0.50 (chance level for a
  model that only knows the marginal distribution of each slot; the held-out colour-shape pairs never appear in
  train, so the blind model may lean very slightly on colour-shape statistics).
* Exact match close to 0 (below 0.03): every part is a coin flip, and there are 3 to 7 of them.
* Teacher-forced letter accuracy is HIGH (>0.7) anyway, because after the first letter of a part
  (`c` -> `ircle`, `g` -> `reen`) the rest is determined by spelling. Letter accuracy *inside parts* should be ~1.0,
  letter accuracy on the *first letter of a part* should be low (~0.3-0.5).

If I am wrong, I will see: (a) attribute accuracy clearly above chance for the blind model, which would mean the
words leak information about themselves (e.g. a colour-size correlation in the generator) or that "zeros" is not
really blind; or (b) the seeing model not clearly beating the blind one, which would mean my pipeline is not
using the image (e.g. a bug in the mask or the adapter).

## Baseline
`configs/baseline.yaml` (sees the image).

## One variable changed
`blind: zeros` in `configs/blind.yaml` (train AND evaluation images are all-zero tensors).
Fixed: architecture, parameter count, optimizer, learning rate, batch size, epochs, seeds 0 1 2, data splits.

## Commands
```
for s in 0 1 2; do python -m src.train --config configs/baseline.yaml --seed $s; done
for s in 0 1 2; do python -m src.train --config configs/blind.yaml --seed $s; done
python experiments/aggregate.py --names baseline blind --split test --out experiments/E1_blind
```

## Result
Test split, mean +- std over seeds 0, 1, 2 (`summary_test.md`, plot in `attribute_accuracy_test.png`):

| metric | baseline | blind |
|---|---|---|
| exact match | 0.717 +- 0.004 | 0.017 +- 0.001 |
| size acc | 0.841 +- 0.004 | 0.332 +- 0.002 |
| color acc | 0.747 +- 0.006 | 0.201 +- 0.000 |
| shape acc | 0.749 +- 0.010 | 0.171 +- 0.031 |
| relation acc | 0.451 +- 0.005 | 0.000 +- 0.000 |
| #objects acc | 1.000 +- 0.000 | 0.490 +- 0.000 |
| letter acc (teacher-forced, all) | 0.988 | 0.874 |
| letter acc, first letter of a part | 0.940 | 0.403 |
| letter acc, inside a part | 1.000 | 0.984 |

## Interpretation
The main model uses the image: the gap in exact match (0.717 vs 0.017) is far larger than the spread over seeds.

My prediction was only partly right. Exact match below 0.03: right. Letters inside a part close to 1.0: right (0.984).
First letter of a part low: right (0.40). Attribute accuracy at chance (size 0.5, others 0.25): wrong. Blind size was
0.33, colour 0.20, shape 0.17, and relation exactly 0. The blind model has the same input for every image, so greedy
decoding gives the same word every time. It only ever writes one object (#objects acc 0.49 is just the share of
one-object images), so relation and the second object can never be right. I predicted chance for a sampling model, but
greedy decoding commits to one word.

Letter accuracy is misleading: 0.874 for a model that is almost never right. Most letters are spelling inside a part.

Not shown by this experiment: only one architecture and setting, three seeds. I did not check whether some
information leaks through the zero image (BatchNorm on constant input), although the near-zero result suggests not.

## Log (failed runs, surprises, dead ends)
The surprise was the blind model writing a single fixed word. The validation loss of the main model has one spike
near step 3400 that recovered by itself.
