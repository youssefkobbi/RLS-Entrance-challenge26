<p align="center">
  <img src="assets/tiny_vlm_logo_rounded.png" alt="Tiny VLM" width="160">
</p>

# Tiny VLM: spelling what a picture shows

My submission for the RLS (Research Lab SUP'COM) entrance challenge. The model looks at a 64x64 image with one or
two coloured shapes and writes a one-word description of it, one letter at a time, for example `largeredcircle` or
`smallbluesquareleftoflargeyellowtriangle`.

<p align="center">
  <img src="assets/Pipeline.png" alt="image, through the tiny VLM, to a word" width="600">
</p>

I did not use any pretrained model. The CNN, the attention, the decoder and the training loop are all written
from scratch in PyTorch. Only `generate_data.py` (the dataset generator) comes from RLS. I used Claude while
working on this, and I describe exactly how in [AI_USAGE.md](AI_USAGE.md).

## Results

Numbers are on the `test` split, mean +- std over seeds. Everything here can be reproduced with the commands in
"How to run".

| Experiment | Configuration | Metric | Result (mean +- std, n seeds) | Interpretation |
|---|---|---|---|---|
| Main model | `configs/baseline.yaml` | Exact match, test | TODO | TODO |
| E1 blind | `configs/blind.yaml` | Attribute acc., test | TODO | TODO |
| S1 throughput | batch 64, TODO (GPU) | images/s | TODO | TODO |

E0 (can the model overfit one batch?) worked: the loss goes from 3.48 to 0.0006 in 400 steps and all 32 words are
reproduced exactly (`experiments/E0_overfit/`, run on my laptop CPU).

## How the model works

1. **CNN encoder.** Three stages of two 3x3 convolutions (with batch norm and ReLU) followed by a 2x2 max-pool.
   The image goes 64 -> 32 -> 16 -> 8 pixels wide, so I end up with an 8x8 grid of 128-channel features.
2. **Adapter.** I flatten the grid into 64 positions and use one linear layer to turn each 128-dim feature into a
   `d_model`-dim vector. These 64 vectors are the "visual tokens".
3. **One sequence.** The visual tokens and the embeddings of the letters written so far are concatenated into a
   single sequence `[v1 ... v64  l1 ... lT]`, and a learned positional embedding is added.
4. **Decoder.** 4 pre-LayerNorm transformer blocks (attention, MLP, residuals) with `d_model=128`, 4 heads.
   The attention is my own implementation in `src/model/attention.py`.
5. **Mask.** The visual tokens can see each other (the whole image is given at once). The letters are causal and can
   also see all the visual tokens. That is how a letter "looks at" the image.
6. **Output.** A linear layer gives 27 scores (26 letters and `<eos>`). The output at the last visual token predicts
   the first letter, the output at letter *i* predicts letter *i+1*, and the last letter predicts `<eos>`.
   At test time I decode greedily until `<eos>` or 45 letters.

The model has about 1.1 million parameters. The full explanation of every tensor shape is in the docstring at the
top of `src/model/vlm.py`.

## How I checked that it is correct

- `tests/test_attention.py` (provided by RLS) compares my attention against `F.scaled_dot_product_attention`, with
  several masks, to about 1e-5.
- `tests/test_shapes.py` (provided) checks shapes, causality and that the logits depend on the image.
- `tests/test_evaluate.py` covers my tokenizer, the padding in the collate function, and the parser that reads
  broken words like `largeredcirle`.
- E0 (overfit one batch) checks that the whole pipeline can learn at all.

## How to run

```bash
pip install -r requirements.txt
python generate_data.py                  # writes data/*.pt, seed 42, I did not change the splits
pytest                                   # all tests

# E0: overfit one batch
python -m src.train --config configs/baseline.yaml --overfit-one-batch

# main model, 3 seeds
for s in 0 1 2; do python -m src.train --config configs/baseline.yaml --seed $s; done

# E1: same model, images replaced by zeros, 3 seeds
for s in 0 1 2; do python -m src.train --config configs/blind.yaml --seed $s; done

# mean +- std table and comparison plot
python experiments/aggregate.py --names baseline blind --split test --out experiments/E1_blind

# S1: training throughput on CPU and GPU
python -m benchmarks.S1_throughput.benchmark
```

Always run from the repository root with `python -m ...`. `--device cpu|cuda|auto` picks the hardware, and
`--resume` continues a run after a Colab disconnect. Training runs write to `runs/<config>_seed<k>/`
(checkpoints, loss curves, `results_test.json`).

## Metrics

- **Exact match:** the generated word equals the reference.
- **Attribute accuracy:** a small parser splits the generated word into size / colour / shape (and relation and
  second object). It matches each part to the closest valid word, so a misspelt word still gets credit for the
  parts that are right.
- **Letter accuracy** (teacher-forced) is reported split into "first letter of a part" and "inside a part". Inside a
  part the next letter is mostly pure spelling (`ircle` after `c`), so this number looks good even for a model that
  ignores the image. That is why I do not use it as the main metric.

## Layout

```
generate_data.py          dataset generator (from RLS, unchanged)
configs/                  baseline.yaml, blind.yaml
src/                      tokenizer, data, model/ (encoder, attention, decoder, vlm), train, generate, evaluate
tests/                    provided tests + my tokenizer/parser tests
experiments/              E0_overfit/, E1_blind/, aggregate.py
benchmarks/S1_throughput/ throughput benchmark, raw CSVs, plots, hardware.txt
report/                   final report
```

## Not done

I focused on Level 1 of the brief. I did not do the Level 2 and 3 items (own experiment E3, profiling breakdown,
memory analysis, held-out error analysis), except that `src.evaluate` can already evaluate on `test_heldout`.
