# AI usage

Describe which AI tools you used (ChatGPT, Claude, Copilot, ...), for which parts of this project, and what
you personally checked, tested, or rewrote. Required for submission — see the Project Brief, section 2.

## Tools used

* **Claude (Anthropic)** — used throughout:
  * to explain the concepts (autograd, CNNs, attention, prefix-LM masking, GPU asynchrony) and to plan the work;
  * to generate the first version of all code in `src/`, `configs/`, `benchmarks/S1_throughput/benchmark.py`,
    `experiments/aggregate.py` and `tests/test_evaluate.py`, and the draft of `experiments/E1_blind/notes.md`;
  * `generate_data.py` and the tests `test_attention.py`, `test_shapes.py`, `test_generate_data.py` were provided by RLS.

## What I personally checked, tested or rewrote

>
> * read every file in `src/` line by line and renamed / commented whatever I would not have written;
> * re-derived the shape of every tensor from image to logits on paper and compared with the printed shapes;
> * hand-checked the attention on a 3-token example and understood why the equivalence test passes;
> * changed one thing on purpose (e.g. removed the `sqrt(d_k)` scaling, flipped the mask) and watched the
>   tests / the loss break;
> * ran E0, the real training, E1 and S1 myself and read the raw logs.

