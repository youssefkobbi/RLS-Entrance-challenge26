<p align="center">
  <img src="assets/tiny_vlm_logo_rounded.png" alt="Tiny VLM" width="200">
</p>

# RLS Entrance Challenge — Tiny Vision-Language Model

RLS (Research Lab SUP'COM) is a student-led lab where members learn research by doing it: reading papers,
building systems, running experiments, and reporting what they find — including what failed.

This repository is the entrance challenge for new members. The task: build a tiny vision-language model that
looks at a generated image and spells out a one-word description of it, letter by letter.

<p align="center">
  <img src="assets/Pipeline.png" alt="RLS input, through the Tiny VLM, to an rls output" width="640">
</p>

## What you're building

```
Image (B, 3, 64, 64)
  -> CNN encoder                         (yours)
  -> visual tokens                       (adapter: flatten + linear projection)
  -> Transformer decoder                 (yours, handwritten multi-head attention)
  -> greedy letter-by-letter decoding
  -> e.g. "largeredcircle"
```

`generate_data.py` in this repository is provided by RLS — run it with the default (fixed) seed and do not
change the split sizes. Everything else — tokenizer, CNN encoder, attention, decoder, training loop,
evaluation, and experiments — is your own implementation.

Full requirements, constraints, rubric, and timeline are in the Learning Guide and Project Brief you were
given. If anything here conflicts with those documents, the documents win.

## Results

Fill in after the final runs (format required by the Project Brief, section 12). Every row is reproducible from
the commands in "Reproduce" below; numbers are on the `test` split, mean +- std over seeds.

| Experiment | Configuration | Metric | Result (mean +- std, n seeds) | Interpretation (1 sentence) |
|---|---|---|---|---|
| Main model | `configs/baseline.yaml` | Exact match, test | TODO | TODO |
| E1 blind | `configs/blind.yaml` | Attribute acc., test | TODO | TODO |
| S1 throughput | batch 64, TODO (GPU name) | images/s | TODO | TODO |

## Getting started

```bash
pip install -r requirements.txt
python generate_data.py            # writes data/{train,val,test,test_heldout}.pt + data/meta.json (seed 42, do not change)
pytest                             # generator, attention, shape, tokenizer and parser tests
python -m src.data                 # prints one decoded sample (tokenizer + dataset check)
```

Run every script from the repository root with `python -m ...` (so `import src...` works).

## Reproduce

```bash
# E0: overfit one batch (sanity check, ~1 min on CPU) -> experiments/E0_overfit/{e0_overfit.png,e0_log.json}
python -m src.train --config configs/baseline.yaml --overfit-one-batch

# Main model: 3 seeds -> runs/baseline_seed{0,1,2}/{best.pt,history.json,loss_curves.png,results_test.json}
for s in 0 1 2; do python -m src.train --config configs/baseline.yaml --seed $s; done

# E1 blind baseline (same model, images replaced by zeros): 3 seeds
for s in 0 1 2; do python -m src.train --config configs/blind.yaml --seed $s; done

# Mean +- std tables and the comparison plot
python experiments/aggregate.py --names baseline blind --split test --out experiments/E1_blind

# Evaluate a checkpoint on other splits / look at predictions
python -m src.evaluate --checkpoint runs/baseline_seed0/best.pt --split test test_heldout
python -m src.generate --checkpoint runs/baseline_seed0/best.pt --split test -n 8

# S1 throughput benchmark (CPU and GPU if available); add --with-loader to include data loading in the timer
python -m benchmarks.S1_throughput.benchmark
```

`--device cpu|cuda|auto` selects the hardware; `--resume` continues a run from `last.pt` after a Colab disconnect.
On Colab: clone the repo, `pip install -r requirements.txt` (keep Colab's own torch build), generate the data once,
and run the commands above -- write `runs/` to Google Drive or push it often.

## Design in one page

| Piece | Choice |
|---|---|
| Tokenizer | ids 0-25 = a-z, 26 = `<eos>`; input = letters, target = letters + `<eos>`; pad input with 0, pad target with -100 (`ignore_index`) |
| Encoder | 3 stages of [Conv3x3-BN-ReLU] x 2 + MaxPool2 : 64 -> 32 -> 16 -> 8; channels 32 / 64 / 128 |
| Adapter | flatten the 8x8 map to 64 positions, `Linear(128, d_model)` -> 64 visual tokens |
| Sequence | `[v1 ... v64  l1 ... lT]` + learned absolute positional embeddings over all 64 + T positions |
| Mask (prefix-LM) | visual tokens attend to each other (bidirectional); letters are causal and see all visual tokens |
| Decoder | 4 pre-LN blocks, `d_model=128`, 4 heads, MLP 512 (GELU), dropout 0.1, handwritten attention |
| Loss position | logit of the last visual token predicts letter 1; logit of letter i predicts letter i+1; last letter predicts `<eos>` |
| Decoding | greedy, until `<eos>` or 45 letters |

## Repository layout

```
.
├── generate_data.py         # RLS-provided ShapeScenes generator — do not modify
├── configs/                 # baseline.yaml, blind.yaml, ...
├── src/
│   ├── tokenizer.py
│   ├── data.py
│   ├── utils.py
│   ├── model/
│   │   ├── encoder.py
│   │   ├── attention.py
│   │   ├── decoder.py
│   │   └── vlm.py
│   ├── train.py
│   ├── evaluate.py
│   └── generate.py
├── tests/
│   ├── test_attention.py       # provided: equivalence with F.scaled_dot_product_attention, masks, gradients
│   ├── test_shapes.py          # provided: tensor shapes and pipeline sanity checks
│   ├── test_generate_data.py   # provided: checks generate_data.py against the spec
│   └── test_evaluate.py        # tokenizer, collate, blind modes and the misspelling-tolerant parser
├── experiments/                # E0_overfit/, E1_blind/, aggregate.py (mean +- std over seeds)
├── benchmarks/S1_throughput/   # benchmark.py, results_*.csv, throughput_*.png, hardware.txt
└── report/
```

## Provided tests

The model tests skip until you implement `src/model/`. They only assume a few conventions, documented at the top
of `tests/test_attention.py` and `tests/test_shapes.py`: the last `nn.Module` in each `src/model/` file is its
main class; attention is `Cls(d_model, n_heads)` called as `attention(x, mask)`, with a boolean mask where `True`
means "may attend"; the encoder and the full model can be built with no arguments.

## AI usage

Document any AI tool usage in `AI_USAGE.md` — required for submission.

## Questions

Ask in the challenge channel — conceptual and requirement questions only, not "please fix my code."

## License

See [LICENSE](LICENSE).
