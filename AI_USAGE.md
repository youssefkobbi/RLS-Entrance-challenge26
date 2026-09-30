# AI usage

Describe which AI tools you used (ChatGPT, Claude, Copilot, ...), for which parts of this project, and what
you personally checked, tested, or rewrote. Required for submission — see the Project Brief, section 2.

## Tools used

* **Claude (Anthropic)** — used throughout:
  * to explain the concepts (autograd, CNNs, attention, prefix-LM masking, GPU asynchrony) and to plan the work;
  * to generate the first version of all code in `src/`, `configs/`, `benchmarks/S1_throughput/benchmark.py`,
    `experiments/aggregate.py` and `tests/test_evaluate.py`, and the draft of `experiments/E1_blind/notes.md`;
  * `generate_data.py` and the tests `test_attention.py`, `test_shapes.py`, `test_generate_data.py` were provided by RLS.

## What I personally did

* Ran E0 on my laptop CPU, then ran the baseline and blind trainings (3 seeds each) on a Colab T4 GPU.
* Ran the S1 throughput benchmark on Colab (GPU and CPU) and on my laptop CPU.
* Ran the aggregation script, looked at the result tables and loss curves, and pushed the repository to GitHub.

## Things I did NOT verify / known gaps

* I did not write the code in `src/` myself; Claude generated the first version. I have not yet checked it line by
  line against my own derivations, and I am studying it before the interview.
* The explanations of the E1 result and of the relation errors are Claude-assisted and are hypotheses, not
  things I have tested.
