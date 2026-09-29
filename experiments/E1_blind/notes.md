# E1 — Blind baseline

> DRAFT written before any E1 run. Youssef: read it, change it to what you actually believe, and commit it BEFORE
> you launch the blind training (the commit timestamp is the evidence that the hypothesis came first).

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
TODO (paste `summary_test.md`, refer to `attribute_accuracy_test.png`).

## Interpretation
TODO: what the result supports, what it does not, and any alternative explanation.
The lesson to draw: the gap between letter accuracy and attribute accuracy for the blind model.

## Log (failed runs, surprises, dead ends)
TODO
