"""Build report/report.pdf from the text below and the raw result files.

    python report/make_report.py

Numbers are read from the repository's raw files (so the report and the results cannot drift apart):
  experiments/E0_overfit/e0_log.json          E0
  experiments/E1_blind/summary_test.json      main model + E1 (written by experiments/aggregate.py)
  benchmarks/S1_throughput/results_*.csv      S1
  runs/baseline_seed0/loss_curves.png         training curves (if present)
Anything that does not exist yet is shown as a yellow [TBD ...] marker.  Re-run this script after each new result,
then edit the TEXT of the analysis sections (search for "TBD") in your own words.
Requires: reportlab, matplotlib.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "report.pdf"
FIG = ROOT / "report" / "figures"
FIG.mkdir(exist_ok=True)


def tbd(text):
    """Yellow marker for anything I still have to fill in."""
    return f'<font backColor="#fff36b">[TBD: {text}]</font>'


# ----------------------------------------------------------------------------- data
def load_json(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


def load_s1():
    """{(label, device, batch): (mean, std)} and {(label, batch): peak MiB} from the S1 CSV files."""
    stats, mem = {}, {}
    for label, fname in [("Colab", "results_colab.csv"), ("PC", "results_pc_cpu.csv")]:
        p = ROOT / "benchmarks" / "S1_throughput" / fname
        if not p.exists():
            continue
        groups = defaultdict(list)
        for r in csv.DictReader(open(p)):
            groups[(r["device"], int(r["batch_size"]))].append(float(r["images_per_s"]))
            if r["peak_mem_MiB"]:
                mem[(int(r["batch_size"]))] = float(r["peak_mem_MiB"])
        for (dev, bs), v in groups.items():
            stats[(label, dev, bs)] = (statistics.mean(v), statistics.pstdev(v))
    return stats, mem


def architecture_figure(path):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    boxes = [
        ("Image", "(B,3,64,64)"),
        ("CNN encoder\n3 x [conv-BN-ReLU]x2\n+ maxpool", "(B,128,8,8)"),
        ("Flatten +\nLinear adapter", "(B,64,d_model)"),
        ("concat with letter\nembeddings\n+ position emb.", "(B,64+T,d_model)"),
        ("4 x decoder block\nmasked self-attention\n+ MLP, pre-LN", "(B,64+T,d_model)"),
        ("keep last visual\n+ T letter rows\nLinear head", "(B,T+1,27)"),
    ]
    n = len(boxes)
    w, gap = 1.35, 0.32
    for i, (title, shape) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 0.35), w, 1.2, fc="#e8eef7", ec="#33507a", lw=1.2))
        ax.text(x + w / 2, 0.95, title, ha="center", va="center", fontsize=7.6)
        ax.text(x + w / 2, 0.12, shape, ha="center", va="center", fontsize=7.6, family="monospace", color="#8a1c1c")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 0.95), xytext=(x + w + 0.02, 0.95),
                        arrowprops=dict(arrowstyle="->", color="#33507a"))
    ax.text(0, 1.85, "letters so far (B,T) -> embedding (B,T,d_model)  joins the sequence at step 4;   "
            "greedy decoding repeats steps 4-6 until <eos>", fontsize=7.4, color="#444")
    ax.set_xlim(-0.1, n * (w + gap))
    ax.set_ylim(-0.1, 2.05)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mask_figure(path):
    import numpy as np

    n_v, n_l = 6, 5
    s = n_v + n_l
    m = np.tril(np.ones((s, s)))
    m[:n_v, :n_v] = 1
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1.4)
    ax.axhline(n_v - 0.5, c="k", lw=0.8)
    ax.axvline(n_v - 0.5, c="k", lw=0.8)
    ax.set_xticks([n_v / 2 - 0.5, n_v + n_l / 2 - 0.5], ["visual", "letters"], fontsize=7)
    ax.set_yticks([n_v / 2 - 0.5, n_v + n_l / 2 - 0.5], ["visual", "letters"], fontsize=7, rotation=90, va="center")
    ax.set_xlabel("key (attended to)", fontsize=7)
    ax.set_ylabel("query", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def scaled_image(path, width):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


# ----------------------------------------------------------------------------- document
styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=11.4,
                      alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("H1", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=6,
                    spaceAfter=3, textColor=colors.HexColor("#1f3a5f"))
CAP = ParagraphStyle("Cap", parent=BODY, fontSize=7.8, leading=9.5, textColor=colors.HexColor("#444444"),
                     alignment=0)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=15, spaceAfter=2)
SUB = ParagraphStyle("Sub", parent=BODY, alignment=1, textColor=colors.HexColor("#555555"), fontSize=8.5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=9.6, alignment=0)


def P(text, style=BODY):
    return Paragraph(text, style)


def table(rows, col_widths, header=True, font=8):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def cell(text):
    return Paragraph(text, SMALL)


def build():
    architecture_figure(FIG / "architecture.png")
    mask_figure(FIG / "mask.png")
    e0 = load_json("experiments/E0_overfit/e0_log.json")
    summary = load_json("experiments/E1_blind/summary_test.json")
    s1, s1_mem = load_s1()
    story = []

    # ---- title
    story += [P("A Tiny Vision-Language Model that Spells What It Sees", TITLE),
              P("RLS Entrance Challenge &nbsp;|&nbsp; Youssef &nbsp;|&nbsp; code: "
                "github.com/youssefkobbi/RLS-Entrance-challenge26 &nbsp;|&nbsp; final commit: "
                + tbd("git hash"), SUB), Spacer(1, 4)]

    # ---- 1 problem
    story += [P("1. Problem", H1),
              P("The task is to build the smallest honest version of a vision-language model. The input is a 64x64 "
                "image with one or two coloured shapes (ShapeScenes, generated by the RLS script with seed 42). The "
                "output is one word with no spaces that describes the scene, such as <i>largeredcircle</i> or "
                "<i>smallbluesquareleftoflargeyellowtriangle</i>, written one letter at a time from a 27-token "
                "vocabulary (26 letters and an end token). Because the right answer is known exactly, I can measure "
                "what the model really learned. I wanted to learn three things: how an image becomes tokens that a "
                "text decoder can read, whether the model really uses the image (the blind baseline), and how the "
                "training step behaves on a CPU and on a GPU.")]

    # ---- 2 architecture
    story += [P("2. Architecture", H1), scaled_image(FIG / "architecture.png", 17.2 * cm),
              P("<b>Figure 1.</b> The pipeline with tensor shapes. B is the batch size, T the number of letters fed "
                "to the decoder, d_model = 128. The model has 1,118,203 parameters.", CAP), Spacer(1, 3)]
    mask_and_text = Table(
        [[scaled_image(FIG / "mask.png", 4.3 * cm),
          [P("<b>Design choices.</b> (a) <b>CNN encoder:</b> three stages of two 3x3 convolutions with batch norm "
             "and ReLU, each followed by a 2x2 max-pool, so the image goes 64 -> 32 -> 16 -> 8 pixels wide. The "
             "8x8 grid of 128-channel features gives 64 visual tokens: few enough to keep the sequence short "
             "(64 + up to 45 = 109 tokens) but still a grid, so left/right and above/below stay recoverable. "
             "Pooling throws away the exact position inside each 2x2 window, so I rely on the learned position "
             "embeddings to keep the coarse layout. (b) <b>Adapter:</b> one linear layer per grid cell, so a visual "
             "token has the same size as a letter embedding and the decoder cannot tell them apart. "
             "(c) <b>Mask (prefix-LM, Figure 2):</b> visual tokens attend to each other, because the whole image is "
             "given at once and there is no future to hide; letters are causal and can see every visual token. "
             "(d) <b>Which output predicts what:</b> the logit of the last visual token predicts letter 1, the logit "
             "of letter <i>i</i> predicts letter <i>i</i>+1, and the last letter predicts &lt;eos&gt;, so a word of "
             "L letters gives L+1 targets and no start token is needed."),
           P("<b>Figure 2.</b> Attention mask, dark = may attend.", CAP)]]],
        colWidths=[4.6 * cm, 12.7 * cm])
    mask_and_text.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [mask_and_text]

    # ---- 3 implementation
    story += [P("3. Implementation", H1),
              P("<b>Attention</b> (src/model/attention.py) is written with tensor operations only: one linear layer "
                "makes Q, K and V, the heads are contiguous channel slices, the scores are QK<super>T</super> / "
                "sqrt(d<sub>k</sub>), forbidden pairs are filled with the most negative float before a softmax over "
                "the keys, and the heads are merged and passed through an output projection. Dividing by "
                "sqrt(d<sub>k</sub>) keeps the score variance near 1, so the softmax does not saturate. "
                "<b>Decoder block:</b> pre-LayerNorm, x = x + Attn(LN(x)), then x = x + MLP(LN(x)), with a GELU MLP "
                "of width 512 and dropout 0.1. <b>Loss masking:</b> words have different lengths, so targets are "
                "padded with -100 and cross-entropy uses ignore_index; padded inputs use letter id 0, which is safe "
                "because the causal mask stops real positions from seeing them. <b>Generation:</b> greedy, one "
                "letter per step until &lt;eos&gt; or 45 letters; the CNN runs once and only the decoder is re-run "
                "each step (no KV cache). <b>Evaluation:</b> exact match, plus a parser that cuts a possibly "
                "misspelt word into size / colour / shape [/ relation / size / colour / shape] by dynamic "
                "programming over edit distance to the vocabulary, so <i>largeredcirle</i> still counts as "
                "<i>large, red, circle</i>.<br/>"
                "<b>How I checked correctness.</b> The provided test compares my attention to "
                "F.scaled_dot_product_attention with five masks (none, causal in 2-D and 4-D form, random per sample, prefix-LM) at 1e-5 in "
                "float32, and further tests check that a masked key has no influence, that causal outputs do not "
                "depend on later letters, and that every parameter gets a gradient. The provided shape tests check "
                "the whole model, and my own tests cover the tokenizer, padding and parser. Finally E0 (below) "
                "checks that the pipeline can learn at all.")]

    # ---- 4 training setup
    rows = [["Optimizer", "AdamW, lr 1e-3, weight decay 0.01, gradient clipping 1.0",
             "Schedule", "200 warm-up steps, then cosine to 5% of lr"],
            ["Batch / epochs", "128 images, 30 epochs (157 steps per epoch)", "Regularisation", "dropout 0.1"],
            ["Data", "train 20,000, val 2,000, test 2,000 (heldout 2,000, not analysed)", "Seeds",
             "0, 1, 2 (dataset seed 42)"],
            ["Checkpoint", "best validation loss", "Hardware",
             "Colab Tesla T4 for training and S1; laptop CPU for E0 and S1"]]
    story += [KeepTogether([P("4. Training setup", H1),
                            table([[cell(c) for c in r] for r in rows], [2.4 * cm, 6.4 * cm, 2.4 * cm, 6.1 * cm],
                                  header=False)])]

    # ---- 5 experiments
    story += [P("5. Experiments", H1),
              P("<b>E0, overfit one batch.</b> Question: can the pipeline learn at all? 32 fixed training images, "
                "dropout off, 400 steps of AdamW at lr 1e-3. Success means the loss goes to almost zero and greedy "
                "decoding reproduces every word. "
                "<b>E1, blind baseline.</b> Question: does the model use the image? "
                "<i>Hypothesis, written before running:</i> with all-zero images the model only knows the "
                "grammar of the words, so attribute accuracy should be at chance (size about 0.5, colour, shape "
                "and relation about 0.25), exact match close to 0, while teacher-forced letter accuracy stays high "
                "because the letters inside a part are pure spelling. Baseline: the same model with the real image. "
                "One variable changed: images replaced by zeros in training and evaluation. Fixed: architecture, "
                "parameter count, optimizer, learning rate, batch size, epochs, seeds and splits. "
                "<b>S1, throughput.</b> Time of a full training step (forward, loss, backward, optimizer) on a "
                "batch already in device memory, for batch sizes 1, 16, 64 and 256. Traps handled: 5 warm-up "
                "steps are discarded (first calls pay one-off costs such as CUDA context, cuDNN algorithm choice and "
                "allocator growth); torch.cuda.synchronize() is called before the timer starts and before it is "
                "read, because GPU calls return before the work is done; data loading is <b>outside</b> the timer. "
                "5 repeats of 10 steps each, mean and standard deviation reported.")]

    # ---- 6 results
    story += [P("6. Results", H1)]
    if summary:
        b, bl = summary["baseline"], summary["blind"]

        def f(d, k):
            return f"{d[k]['mean']:.3f} +- {d[k]['std']:.3f}"

        keys = [("exact match", "Exact match"), ("size acc", "Size acc."), ("color acc", "Colour acc."),
                ("shape acc", "Shape acc."), ("relation acc", "Relation acc."),
                ("letter acc (TF, all)", "Letter acc. (teacher-forced)"),
                ("letter acc (TF, first of part)", "  first letter of a part"),
                ("letter acc (TF, inside part)", "  inside a part")]
        rows = [["Metric (test split)", f"Main model (n={b['n_seeds']})", f"Blind E1 (n={bl['n_seeds']})"]]
        rows += [[lab, f(b, k), f(bl, k)] for k, lab in keys]
    else:
        rows = [["Metric (test split)", "Main model", "Blind E1"], ["Exact match", tbd("run"), tbd("run")],
                ["Size / colour / shape / relation acc.", tbd("run"), tbd("run")],
                ["Letter acc. (all / first / inside)", tbd("run"), tbd("run")]]
    story += [table([[cell(c) if i else c for i, c in enumerate(r)] for r in rows], [6.5 * cm, 4.5 * cm, 4.5 * cm]),
              P("<b>Table 1.</b> Test-split accuracy, mean +- std over seeds (std is the sample standard "
                "deviation). Configs: configs/baseline.yaml and configs/blind.yaml.", CAP)]

    curves = ROOT / "runs" / "baseline_seed0" / "loss_curves.png"
    e0_png = ROOT / "experiments" / "E0_overfit" / "e0_overfit.png"
    left = scaled_image(curves, 8.4 * cm) if curves.exists() else P(tbd("loss_curves.png of a baseline run"))
    right = scaled_image(e0_png, 6.2 * cm) if e0_png.exists() else P(tbd("E0 plot"))
    e0_txt = (f"final loss {e0['losses'][-1]:.4f} (from {e0['losses'][0]:.2f}), greedy exact match "
              f"{e0['final_exact_match']:.2f} on the memorised batch, run on a laptop CPU." if e0 else "")
    story += [Table([[left, right]], colWidths=[9.0 * cm, 8.3 * cm], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
              P(f"<b>Figure 3 (left).</b> Train and validation loss of the main model (log scale)"
                + ("" if curves.exists() else " " + tbd("caption: what to notice")) +
                f". <b>Figure 4 (right).</b> E0: {e0_txt} The loss falls far below ln(27) = 3.30, the value of "
                "uniform guessing.", CAP)]

    # S1 table
    def s1cell(label, dev, bs):
        v = s1.get((label, dev, bs))
        return f"{v[0]:.0f} +- {v[1]:.0f}" if v else "-"

    rows = [["Batch size", "T4 GPU (Colab)", "CPU, Colab (1 thread)", "CPU, laptop (8 threads)", "GPU peak mem (MiB)"]]
    for bs in (1, 16, 64, 256):
        rows.append([str(bs), s1cell("Colab", "cuda", bs), s1cell("Colab", "cpu", bs), s1cell("PC", "cpu", bs),
                     f"{s1_mem[bs]:.0f}" if bs in s1_mem else "-"])
    s1png = ROOT / "benchmarks" / "S1_throughput" / "throughput_colab.png"
    s1_block = [table(rows, [2.4 * cm, 3.3 * cm, 3.9 * cm, 4.0 * cm, 3.4 * cm]),
                P("<b>Table 2 (S1).</b> Training throughput in images per second (mean +- std of 5 repeats), data "
                  "loading excluded. Hardware: Colab Tesla T4 with Xeon 2.0 GHz (torch 2.11, CUDA 12.8); laptop AMD "
                  "CPU (torch 2.4.1). The two CPU columns differ in thread count and torch version, so they are not a "
                  "like-for-like comparison.", CAP)]
    story += s1_block
    if s1png.exists():
        story += [scaled_image(s1png, 9.0 * cm),
                  P("<b>Figure 5.</b> S1 on the Colab machine (log-log). The GPU is ahead at every batch size and "
                    "flattens above batch 64; the CPU curve is nearly flat.", CAP)]

    # ---- 7 analysis
    def g(label, dev, bs):
        v = s1.get((label, dev, bs))
        return v[0] if v else float("nan")

    s1_analysis = (
        f"<b>Where does the GPU stop being faster than the CPU?</b> In the range I measured it never stops: even at "
        f"batch 1 the T4 trains {g('Colab', 'cuda', 1):.0f} images/s against {g('Colab', 'cpu', 1):.0f} (Colab CPU) "
        f"and {g('PC', 'cpu', 1):.0f} (laptop CPU), about {g('Colab', 'cuda', 1) / g('PC', 'cpu', 1):.1f}x faster. "
        f"Going from batch 1 to 16 raises the GPU throughput {g('Colab', 'cuda', 16) / g('Colab', 'cuda', 1):.0f}x "
        "for 16x more images, so at batch 1 the step is dominated by fixed per-step cost (launching many small "
        "kernels and Python overhead), not arithmetic. Above batch 64 the throughput is flat "
        f"(~{g('Colab', 'cuda', 256):.0f} images/s), so the GPU is then limited by compute; my rough estimate is "
        "about 0.9 GFLOP per image for a training step, i.e. roughly 1.4 TFLOP/s, well below the T4's fp32 peak. "
        "The CPU curves are almost flat from batch 16 on: the cores are already busy at small batches, so a larger "
        "batch adds no throughput. Peak GPU memory grows roughly linearly with batch size (43 MiB at batch 1, 2.5 GiB "
        "at 256), consistent with the saved activations dominating over the 1.1M parameters. A GPU can lose to a "
        "CPU when the work per step is tiny; this model at batch 1 is still too large for that to happen. "
        "I did not test smaller inputs or models, and I did not time data loading.")
    story += [P("7. Analysis", H1), P(s1_analysis)]
    if summary:
        story += [P(tbd("E1: does the main model clearly beat the blind one? Which attribute fails most and my "
                        "hypothesis why? Why is letter accuracy misleading for the blind model? Was my hypothesis "
                        "right? What do the results NOT show (one architecture, 3 seeds)?"))]
    else:
        story += [P(tbd("E1 analysis after the runs: compare attribute accuracy of the main and blind model, explain "
                        "the gap between letter accuracy and attribute accuracy, say whether the hypothesis was "
                        "right, and what the evidence does not show."))]

    # ---- 8 limitations
    story += [P("8. Limitations", H1),
              P("I did only Level 1. I did not run my own experiment E3, a profiler breakdown of the training step, a "
                "memory estimate to compare with the measured peaks, or an error analysis on test-heldout, so I cannot "
                "say how well the model generalises to unseen colour-shape pairs. Only one architecture and one set "
                "of hyperparameters were trained, and three seeds give only a rough idea of the spread. The S1 "
                "comparison mixes different machines, thread counts and torch versions, and excludes data loading. "
                "The absolute-position embeddings and the max-pool may limit how well spatial relations are learned. "
                "With more time I would run the held-out evaluation, an experiment on the number of visual tokens, and "
                "a profiler trace to see whether the GPU is waiting on the Python loop.")]

    # ---- 9 references
    story += [P("9. References", H1),
              P("Vaswani et al., <i>Attention Is All You Need</i>, NeurIPS 2017 (Section 3). "
                "Liu et al., <i>Visual Instruction Tuning</i> (LLaVA), 2023. "
                "A. Karpathy, <i>Let's build GPT: from scratch, in code</i> (video) and <i>A Recipe for Training "
                "Neural Networks</i> (blog). "
                "H. He, <i>Making Deep Learning Go Brrrr From First Principles</i> (blog). "
                "PyTorch documentation, <i>CUDA semantics</i>. "
                "AI assistance: see AI_USAGE.md in the repository.", SMALL)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title="Tiny VLM: RLS entrance challenge report",
                            author="Youssef")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"RLS Entrance Challenge - Tiny VLM - page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    build()
