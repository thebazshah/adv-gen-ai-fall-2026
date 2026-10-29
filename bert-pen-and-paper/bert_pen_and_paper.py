#!/usr/bin/env python3
"""Pen-and-paper BERT: generates MLM.md and NSP.md in this folder.

Both documents use the *same* tiny BERT (same embeddings and encoder layer);
only the pre-training head on top differs, exactly as in real BERT, where the
two heads are trained jointly on one shared encoder.

Every number in the documents is produced by this script. Each intermediate
result is rounded to 4 decimals and the *rounded* value is what the next step
uses, so the documents can be reproduced with a pocket calculator. Run

    python3 bert_pen_and_paper.py          # regenerate MLM.md and NSP.md
    python3 bert_pen_and_paper.py --check  # also compare to full precision

Only numpy is required.
"""

import argparse
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Hyperparameters, vocabulary and the two training examples
# --------------------------------------------------------------------------
VOCAB = ["[CLS]", "[SEP]", "[MASK]", "socrates", "taught", "plato", "wrote", "met"]
D_MODEL, N_HEADS, D_FF, MAX_LEN = 4, 2, 8, 8
D_K = D_MODEL // N_HEADS
DECIMALS = 4
GELU_C = 0.044715

# Masked language modelling: "socrates taught plato", "taught" is masked.
MLM_ORIGINAL = ["[CLS]", "socrates", "taught", "plato", "[SEP]"]
MLM_MASK_POS = 2
MLM_TOKENS = [t if i != MLM_MASK_POS else "[MASK]" for i, t in enumerate(MLM_ORIGINAL)]
MLM_SEGMENTS = [0] * len(MLM_TOKENS)

# Next sentence prediction: A = "socrates taught plato", B = "plato wrote".
NSP_TOKENS = ["[CLS]", "socrates", "taught", "plato", "[SEP]", "plato", "wrote", "[SEP]"]
NSP_SEGMENTS = [0, 0, 0, 0, 0, 1, 1, 1]
NSP_CLASSES = ["IsNext", "NotNext"]
NSP_LABEL = 0  # B really follows A

# --------------------------------------------------------------------------
# Parameters (one decimal each so that hand multiplication stays easy)
# --------------------------------------------------------------------------
TOK = np.array([  # token embeddings, also the (tied) MLM output matrix
    [-0.2, -0.4, 0.5, -0.2],
    [0.6, 0.2, -0.8, 0.0],
    [-0.2, 0.1, 0.2, 0.1],
    [0.2, 0.8, -0.3, -0.9],
    [0.4, 0.5, -0.1, 0.7],
    [-0.5, 0.4, 0.3, -0.2],
    [0.6, 0.9, -0.5, -1.0],
    [0.3, 0.6, -0.8, 0.2],
])
SEG = np.array([  # segment (token type) embeddings: row 0 = A, row 1 = B
    [0.5, -1.0, 0.3, -0.4],
    [-0.8, -0.3, 0.1, -0.7],
])
POS = np.array([  # learned position embeddings
    [-0.4, 0.3, -0.9, -0.1],
    [0.7, 0.5, 0.8, -0.3],
    [1.0, -0.9, -0.2, 0.5],
    [0.7, 0.5, 0.5, -0.9],
    [-0.2, -0.1, 0.5, -0.1],
    [0.3, 0.9, 0.1, 0.2],
    [0.0, -0.4, -0.5, 0.8],
    [-0.3, -0.4, -0.2, 0.5],
])
ATTN = {
    "Q": np.array([
        [0.4, -0.7, -1.0, -0.6],
        [-1.0, -0.7, -0.1, 0.7],
        [0.5, -0.6, 0.4, -0.5],
        [-0.1, -0.6, 0.6, 0.6],
    ]),
    "K": np.array([
        [0.5, 0.0, 0.6, -0.6],
        [0.8, 0.9, 1.0, 0.7],
        [0.0, -0.4, -0.5, -0.4],
        [-0.3, -0.8, 0.3, -0.8],
    ]),
    "V": np.array([
        [0.6, 0.5, 0.6, 0.7],
        [0.6, 0.4, -0.5, -0.5],
        [-0.8, 0.7, 1.0, -0.5],
        [-0.5, -0.5, 0.9, 0.3],
    ]),
    "O": np.array([
        [-0.2, -0.5, -0.2, 0.6],
        [-0.2, -0.4, -0.6, 0.7],
        [0.1, 0.7, 0.2, 0.2],
        [-0.4, 0.8, -0.8, 0.3],
    ]),
}
FFN = {
    "W1": np.array([
        [0.6, -0.3, -0.2, -0.5, 0.6, 0.6, 0.7, -0.8],
        [-1.0, 0.5, -0.8, 0.1, -0.7, -1.0, 0.3, -0.6],
        [0.5, 0.0, 0.0, -0.6, 1.0, 0.4, -0.1, 0.5],
        [1.0, 1.0, 0.9, 0.8, -0.9, 0.3, -0.2, 0.3],
    ]),
    "b1": np.array([-0.5, 0.4, 0.2, -0.8, 1.0, -0.9, -0.8, 0.8]),
    "W2": np.array([
        [-0.9, 0.7, -0.4, -0.1],
        [0.5, -0.6, -0.8, -0.9],
        [0.1, 0.6, 0.7, -0.4],
        [-0.6, 0.2, 0.0, -0.2],
        [0.6, -0.1, -0.2, -0.1],
        [-1.0, -0.4, -0.7, 0.2],
        [0.4, 0.1, -0.9, -0.7],
        [-0.8, 0.5, 0.5, 0.7],
    ]),
    "b2": np.array([-0.7, 0.1, -1.0, 0.4]),
}
MLM_HEAD = {
    "Wt": np.array([  # "transform" dense layer
        [0.2, -0.5, 0.9, -1.0],
        [0.7, 0.3, -0.5, 0.5],
        [-0.2, 0.7, 1.0, -0.5],
        [-0.1, -0.6, 0.4, 0.3],
    ]),
    "bt": np.array([0.7, 0.6, 1.0, 1.0]),
    "bv": np.array([0.7, 0.3, -1.0, -0.3, 0.2, 0.5, -0.5, 0.8]),  # output bias, one per word
}
NSP_HEAD = {
    "Wp": np.array([  # pooler
        [0.5, -0.3, -1.0, -0.2],
        [0.0, -0.6, 1.0, -0.8],
        [-0.7, 0.1, 0.5, 0.8],
        [0.3, 0.6, -0.8, -0.2],
    ]),
    "bp": np.array([0.9, -0.1, -0.3, -0.1]),
    "Wc": np.array([  # classifier
        [0.5, 0.1],
        [0.0, -1.0],
        [-0.3, 0.2],
        [0.1, -0.6],
    ]),
    "bc": np.array([-0.8, -0.7]),
}


# --------------------------------------------------------------------------
# The model. `rnd` is applied after every step: rounding for the hand-
# calculation version, identity for the full-precision reference.
# --------------------------------------------------------------------------
def make_rounder(full_precision):
    if full_precision:
        return lambda x: np.asarray(x, dtype=float)
    return lambda x: np.round(np.asarray(x, dtype=float), DECIMALS) + 0.0  # +0.0 drops "-0"


def softmax(x, rnd):
    x = np.atleast_2d(x)
    exps = rnd(np.exp(x))
    sums = rnd(exps.sum(axis=-1))
    return exps, sums, rnd(exps / sums[:, None])


def layer_norm(x, rnd):
    x = np.atleast_2d(x)
    mu = rnd(x.mean(axis=1))
    var = rnd(((x - mu[:, None]) ** 2).mean(axis=1))
    sd = rnd(np.sqrt(var))
    return {"inp": x, "mu": mu, "var": var, "sd": sd, "out": rnd((x - mu[:, None]) / sd[:, None])}


def gelu(x, rnd):
    """tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
    c = rnd(np.sqrt(2 / np.pi))
    a = rnd(x + GELU_C * x ** 3)
    inner = rnd(c * a)
    t = rnd(np.tanh(inner))
    return {"c": c, "a": a, "inner": inner, "tanh": t, "out": rnd(0.5 * x * (1 + t))}


def multi_head_attention(x, rnd):
    Q, K, V = rnd(x @ ATTN["Q"]), rnd(x @ ATTN["K"]), rnd(x @ ATTN["V"])
    scale = rnd(np.sqrt(D_K))
    heads = []
    for h in range(N_HEADS):
        cols = slice(h * D_K, (h + 1) * D_K)
        q, k, v = Q[:, cols], K[:, cols], V[:, cols]
        scores = rnd(q @ k.T)
        scaled = rnd(scores / scale)
        exps, sums, A = softmax(scaled, rnd)
        heads.append({"q": q, "k": k, "v": v, "scores": scores, "scaled": scaled,
                      "exps": exps, "sums": sums, "A": A, "Z": rnd(A @ v)})
    concat = np.hstack([hd["Z"] for hd in heads])
    return {"Q": Q, "K": K, "V": V, "scale": scale, "heads": heads,
            "concat": concat, "out": rnd(concat @ ATTN["O"])}


def encoder(tokens, segments, rnd):
    ids = [VOCAB.index(t) for t in tokens]
    S = {"ids": ids, "tok": TOK[ids], "seg": SEG[segments], "pos": POS[:len(ids)]}
    S["sum"] = rnd(S["tok"] + S["seg"] + S["pos"])
    S["emb_ln"] = layer_norm(S["sum"], rnd)
    X = S["emb_ln"]["out"]
    S["attn"] = multi_head_attention(X, rnd)
    S["ln1"] = layer_norm(rnd(X + S["attn"]["out"]), rnd)
    X1 = S["ln1"]["out"]
    pre = rnd(X1 @ FFN["W1"] + FFN["b1"])
    act = gelu(pre, rnd)
    S["ffn"] = {"pre": pre, "gelu": act, "out": rnd(act["out"] @ FFN["W2"] + FFN["b2"])}
    S["ln2"] = layer_norm(rnd(X1 + S["ffn"]["out"]), rnd)
    S["out"] = S["ln2"]["out"]
    return S


def mlm_forward(rnd):
    S = encoder(MLM_TOKENS, MLM_SEGMENTS, rnd)
    h = S["out"][MLM_MASK_POS]
    pre = rnd(h @ MLM_HEAD["Wt"] + MLM_HEAD["bt"])
    act = gelu(pre, rnd)
    ln = layer_norm(act["out"], rnd)
    logits = rnd(ln["out"][0] @ TOK.T + MLM_HEAD["bv"])
    exps, sums, probs = softmax(logits, rnd)
    target = VOCAB.index(MLM_ORIGINAL[MLM_MASK_POS])
    S["head"] = {"h": h, "pre": pre, "gelu": act, "ln": ln, "logits": logits,
                 "exps": exps[0], "sum": sums[0], "probs": probs[0], "target": target,
                 "loss": rnd(-np.log(probs[0, target]))}
    return S


def nsp_forward(rnd):
    S = encoder(NSP_TOKENS, NSP_SEGMENTS, rnd)
    c = S["out"][0]
    pre = rnd(c @ NSP_HEAD["Wp"] + NSP_HEAD["bp"])
    pooled = rnd(np.tanh(pre))
    logits = rnd(pooled @ NSP_HEAD["Wc"] + NSP_HEAD["bc"])
    exps, sums, probs = softmax(logits, rnd)
    S["head"] = {"c": c, "pre": pre, "pooled": pooled, "logits": logits, "exps": exps[0],
                 "sum": sums[0], "probs": probs[0], "loss": rnd(-np.log(probs[0, NSP_LABEL]))}
    return S


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------
def f(x, d=DECIMALS):
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def fm(x, d=DECIMALS):
    """Number for LaTeX: negatives in parentheses so products read clearly."""
    s = f(x, d)
    return f"({s})" if s.startswith("-") else s


def table(M, rows, cols, d=DECIMALS):
    M = np.atleast_2d(M)
    lines = ["| | " + " | ".join(cols) + " |", "|:--|" + "--:|" * len(cols)]
    for label, row in zip(rows, M):
        lines.append(f"| **{label}** | " + " | ".join(f(x, d) for x in row) + " |")
    return "\n".join(lines)


def math(*lines):
    return "```math\n" + "\n".join(lines) + "\n```"


def aligned(lines):
    return math(r"\begin{aligned}", " \\\\\n".join(lines), r"\end{aligned}")


def dot(a, b, da=DECIMALS, db=DECIMALS):
    return " + ".join(f"{fm(x, da)}\\cdot {fm(y, db)}" for x, y in zip(a, b))


def vec(v, d=DECIMALS):
    return r"\begin{bmatrix}" + " & ".join(f(x, d) for x in v) + r"\end{bmatrix}"


def labels(tokens):
    return [f"{i}: {t}" for i, t in enumerate(tokens)]


DIMS = [f"dim {j}" for j in range(D_MODEL)]
HEAD_DIMS = [f"dim {j} (head {j // D_K + 1})" for j in range(D_MODEL)]
HIDDEN = [f"u{j}" for j in range(D_FF)]
W_ROWS = [f"in {j}" for j in range(D_MODEL)]


# --------------------------------------------------------------------------
# Document sections shared by MLM.md and NSP.md
# --------------------------------------------------------------------------
def doc_setup(md, task_rows):
    A = md.append
    A("### 0.1 What BERT is\n")
    A("BERT (Devlin et al., 2019, *“BERT: Pre-training of Deep Bidirectional Transformers for "
      "Language Understanding”*) is the **encoder** of the Transformer, without the decoder. "
      "Because there is no decoder there is **no causal mask**: every token attends to every "
      "other token, to its left *and* to its right (“bidirectional”). BERT is pre-trained with "
      "two self-supervised tasks that share one encoder and differ only in the small head on top:\n")
    A("| Task | Input | Read the output at | Head | Predicts |\n|:--|:--|:--|:--|:--|\n"
      "| **MLM** — masked language modelling | one sentence with ~15 % of the words hidden | each "
      "masked position | dense + GELU + LayerNorm, then $`\\cdot E^{\\top} + b`$ | the hidden word "
      "(softmax over the vocabulary) |\n"
      "| **NSP** — next sentence prediction | a sentence pair `[CLS] A [SEP] B [SEP]` | the `[CLS]` "
      "position | pooler (dense + tanh), then a 2-way classifier | IsNext / NotNext |\n")
    A("During pre-training both losses are computed on the same batch and added: "
      "$`\\mathcal L = \\mathcal L_{\\text{MLM}} + \\mathcal L_{\\text{NSP}}`$. "
      "The companion document covers the other head; the embedding and encoder weights are "
      "identical in both.\n")
    A("### 0.2 Hyperparameters\n")
    A("| Symbol | Meaning | BERT-base | This example |\n|:--|:--|--:|--:|\n"
      "| $`d_{\\text{model}}`$ | hidden size | 768 | **4** |\n"
      "| $`h`$ | attention heads | 12 | **2** |\n"
      "| $`d_k = d_{\\text{model}}/h`$ | width of one head | 64 | **2** |\n"
      "| $`d_{ff}`$ | feed-forward hidden size | 3072 | **8** |\n"
      "| $`L`$ | encoder layers | 12 | **1** |\n"
      "| — | maximum sequence length | 512 | **8** |\n"
      "| $`\\lvert V\\rvert`$ | vocabulary (WordPiece) | 30 522 | **8** |\n")
    A("Simplifications that do not change the mechanics: no dropout; the Q/K/V/O projections "
      "have no bias (real BERT has them); every LayerNorm uses $`\\gamma = 1`$, $`\\beta = 0`$, "
      "$`\\epsilon \\approx 0`$; whole words instead of WordPiece sub-words. All weights are "
      "made-up “untrained” values with one decimal.\n")
    A("### 0.3 Vocabulary\n")
    A("| id | " + " | ".join(str(i) for i in range(len(VOCAB))) + " |\n|:--|"
      + ":-:|" * len(VOCAB) + "\n| **token** | " + " | ".join(f"`{t}`" for t in VOCAB) + " |\n")
    A("`[CLS]` (classification) always starts the input, `[SEP]` (separator) ends each "
      "sentence, `[MASK]` replaces a hidden word in MLM.\n")
    A("### 0.4 This example\n")
    A(task_rows)
    A("### 0.5 Conventions\n")
    A("* **Row vectors.** Each token is a row; a sequence of $`n`$ tokens is an $`n \\times 4`$ "
      "matrix, and a linear layer is $`XW + b`$.\n"
      "* Positions, dimensions and ids are counted from 0; heads from 1. In formulas the first "
      "subscript is the **position** (e.g. $`q_{2,1}`$ = query of position 2, dimension 1; "
      "$`s_{2,4}`$ = score of query 2 against key 4).\n"
      "* Every intermediate result is rounded to **4 decimals** and the rounded value is used "
      "in the next step.\n"
      "* Weight tables: rows = input dimension (“in 0…3”), columns = output dimension.\n")


def doc_layer_norm(md, ln, toks, wr, out_name, in_desc, a_mat=None, b_mat=None):
    md.append(math(rf"{out_name} = \operatorname{{LayerNorm}}({in_desc}),\qquad "
                   r"\operatorname{LayerNorm}(r) = \gamma \odot \frac{r - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta"))
    md.append("\n$`\\mu`$ and $`\\sigma^2`$ are the mean and (population) variance of the 4 entries "
              "**of each row** — every token is normalised on its own.\n")
    r, mu, var, sd = ln["inp"][wr], ln["mu"][wr], ln["var"][wr], ln["sd"][wr]
    md.append(f"Worked row **{toks[wr]}**:\n")
    lines = []
    if a_mat is not None:
        lines.append(rf"r &= {vec(a_mat[wr])} + {vec(b_mat[wr])} = {vec(r)}")
    else:
        lines.append(rf"r &= {vec(r)}")
    lines += [
        rf"\mu &= \tfrac14({' + '.join(fm(x) for x in r)}) = {f(mu)}",
        rf"\sigma^2 &= \tfrac14\left({' + '.join(f'{fm(x - mu)}^2' for x in r)}\right) = {f(var)}",
        rf"\sigma &= \sqrt{{{f(var)}}} = {f(sd)}",
        rf"\hat r &= \frac{{r - {fm(mu)}}}{{{f(sd)}}} = {vec(ln['out'][wr])}",
    ]
    md.append(aligned(lines))
    if len(toks) > 1:
        md.append("\n| | μ | σ² | σ |\n|:--|--:|--:|--:|")
        md.append("\n".join(f"| **{t}** | {f(ln['mu'][i])} | {f(ln['var'][i])} | {f(ln['sd'][i])} |"
                            for i, t in enumerate(toks)))
        md.append(f"\n$`{out_name}`$ (every row now has mean 0 and variance 1):\n")
        md.append(table(ln["out"], toks, DIMS))
    md.append("")


def doc_gelu_row(md, act, j_labels, row=None):
    """Table of the GELU steps for one row of pre-activations."""
    pick = (lambda k: act[k][row]) if row is not None else (lambda k: act[k])
    md.append("| | " + " | ".join(j_labels) + " |\n|:--|" + "--:|" * len(j_labels))
    md.append(f"| $`x`$ | " + " | ".join(f(v) for v in pick("x")) + " |")
    md.append(f"| $`a = x + 0.044715x^3`$ | " + " | ".join(f(v) for v in pick("a")) + " |")
    md.append(f"| $`{f(act['c'])}\\,a`$ | " + " | ".join(f(v) for v in pick("inner")) + " |")
    md.append(f"| $`\\tanh(\\cdot)`$ | " + " | ".join(f(v) for v in pick("tanh")) + " |")
    md.append(f"| $`\\text{{GELU}} = \\tfrac12 x(1+\\tanh)`$ | "
              + " | ".join(f(v) for v in pick("out")) + " |")


def doc_input(md, S, tokens, segments, wr, extra_rows=()):
    A = md.append
    toks = labels(tokens)
    A("BERT's input vector for each token is the **sum of three learned embeddings**, followed by "
      "a LayerNorm:\n")
    A(math(r"x_i = \operatorname{LayerNorm}\big(E^{\text{tok}}[\text{token}_i] + "
           r"E^{\text{seg}}[\text{segment}_i] + E^{\text{pos}}[i]\big)"))
    A("\n* **Token embedding** $`E^{\\text{tok}}`$ (8×4): one row per vocabulary entry.\n"
      "* **Segment embedding** $`E^{\\text{seg}}`$ (2×4): row A for tokens of the first sentence "
      "(including `[CLS]` and the first `[SEP]`), row B for the second sentence.\n"
      "* **Position embedding** $`E^{\\text{pos}}`$ (8×4): unlike the original Transformer's fixed "
      "sinusoids, BERT *learns* one vector per position (up to 512).\n"
      "* There is no $`\\sqrt{d_{\\text{model}}}`$ scaling.\n")
    A("**Parameters**\n")
    A("$`E^{\\text{tok}}`$\n")
    A(table(TOK, [f"{i}: {t}" for i, t in enumerate(VOCAB)], DIMS, d=1))
    A("\n$`E^{\\text{seg}}`$\n")
    A(table(SEG, ["A (0)", "B (1)"], DIMS, d=1))
    A("\n$`E^{\\text{pos}}`$\n")
    A(table(POS, [f"pos {i}" for i in range(MAX_LEN)], DIMS, d=1))
    A("\n#### Step 1 — look up the three embeddings\n")
    A("| pos | token | id | segment |\n|:-:|:--|:-:|:-:|")
    A("\n".join(f"| {i} | `{t}` | {S['ids'][i]} | {'AB'[s]} |" for i, (t, s) in enumerate(zip(tokens, segments))))
    A("\nToken embeddings (row `id` of $`E^{\\text{tok}}`$):\n")
    A(table(S["tok"], toks, DIMS, d=1))
    A("\nSegment embeddings:\n")
    A(table(S["seg"], toks, DIMS, d=1))
    A("\nPosition embeddings (first rows of $`E^{\\text{pos}}`$):\n")
    A(table(S["pos"], toks, DIMS, d=1))
    A("\n#### Step 2 — add them\n")
    lines = [rf"\text{{pos }}{i}\;(\texttt{{{tokens[i]}}}):\;& {vec(S['tok'][i], 1)} + {vec(S['seg'][i], 1)} + "
             rf"{vec(S['pos'][i], 1)} = {vec(S['sum'][i])}" for i in (wr, *extra_rows)]
    A(aligned(lines))
    A("\n**Embedding sum**\n")
    A(table(S["sum"], toks, DIMS))
    A("\n#### Step 3 — embedding LayerNorm\n")
    doc_layer_norm(md, S["emb_ln"], toks, wr, "X", r"E^{\text{tok}} + E^{\text{seg}} + E^{\text{pos}}")


def doc_attention(md, sec, S, tokens, wr):
    A = md.append
    toks = labels(tokens)
    att, X = S["attn"], S["emb_ln"]["out"]
    n = len(tokens)
    A(math(r"\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \text{head}_2)\,W^O,"
           r"\qquad \text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i"))
    A("\nQueries, keys and values all come from $`X`$ (**self**-attention) and there is **no mask**: "
      "the score matrix is full, so position "
      f"{wr} sees the tokens before *and* after it.\n")
    A(f"#### {sec}.1 Parameters\n")
    for key in "QKVO":
        A(f"$`W^{key}`$ (4×4)\n")
        A(table(ATTN[key], W_ROWS, DIMS if key == "O" else HEAD_DIMS, d=1))
        A("")
    A(f"#### {sec}.2 Queries, keys and values\n")
    A(math(r"Q = X W^Q,\qquad K = X W^K,\qquad V = X W^V"))
    A(f"\nEach is {n}×4. Worked row **{toks[wr]}** (row {wr} of $`X`$ times each column):\n")
    lines = [rf"{m.lower()}_{{{wr},{j}}} &= {dot(X[wr], ATTN[m][:, j], db=1)} = {f(att[m][wr, j])}"
             for m in "QKV" for j in range(D_MODEL)]
    A(aligned(lines))
    for m in "QKV":
        A(f"\n**{m}**\n")
        A(table(att[m], toks, HEAD_DIMS))
    A("")
    A(f"#### {sec}.3 Split into {N_HEADS} heads\n")
    A("Head 1 uses columns 0–1 of $`Q, K, V`$ and head 2 uses columns 2–3, so each head works "
      f"in its own $`d_k = {D_K}`$-dimensional subspace; scores are divided by "
      f"$`\\sqrt{{d_k}} = \\sqrt 2 \\approx {f(att['scale'])}`$.\n")
    for h, H in enumerate(att["heads"], start=1):
        A(f"#### {sec}.{3 + h} Head {h}\n")
        A(f"$`Q_{h}, K_{h}, V_{h}`$ (columns {2 * h - 2}–{2 * h - 1}):\n")
        A(table(np.hstack([H["q"], H["k"], H["v"]]), toks,
                [f"{m}{h}[{c}]" for m in "QKV" for c in range(D_K)]))
        A(f"\n**Step 1 – scores** $`S_{h} = Q_{h}K_{h}^{{\\top}}`$ ({n}×{n}). Worked row "
          f"**{toks[wr]}** (query {wr} against every key $`j`$):\n")
        A(aligned([rf"s_{{{wr},{j}}} &= {dot(H['q'][wr], H['k'][j])} = {f(H['scores'][wr, j])}"
                   for j in range(n)]))
        A("")
        A(table(H["scores"], toks, toks))
        A(f"\n**Step 2 – scale** by $`{f(att['scale'])}`$ (e.g. "
          f"$`{fm(H['scores'][wr, 0])} / {f(att['scale'])} = {f(H['scaled'][wr, 0])}`$):\n")
        A(table(H["scaled"], toks, toks))
        A(f"\n**Step 3 – softmax of each row.** Worked row **{toks[wr]}**:\n")
        terms = [rf"e^{{{f(H['scaled'][wr, j])}}} = {f(H['exps'][wr, j])}" for j in range(n)]
        lines = [r"&" + r",\quad ".join(terms[k:k + 4]) for k in range(0, n, 4)]
        lines += [rf"&\textstyle\sum = " + " + ".join(f(x) for x in H["exps"][wr]) + f" = {f(H['sums'][wr])}",
                  rf"&a_{{{wr},\cdot}} = \frac{{1}}{{{f(H['sums'][wr])}}}\,{vec(H['exps'][wr])} = {vec(H['A'][wr])}"]
        A(aligned(lines))
        A(f"\n**Attention weights** $`A_{h}`$ (row = query, column = key; rows sum to 1 up to rounding):\n")
        A(table(H["A"], toks, toks))
        best = [f"{toks[i]} → **{toks[int(np.argmax(H['A'][i]))]}** ({f(H['A'][i].max())})" for i in range(n)]
        A("\nStrongest link per query: " + "; ".join(best) + ".")
        A(f"\n**Step 4 – weighted sum of values** $`Z_{h} = A_{h}V_{h}`$ ({n}×2). Worked row **{toks[wr]}**:\n")
        A(aligned([rf"z_{{{wr},{c}}} &= {dot(H['A'][wr], H['v'][:, c])} = {f(H['Z'][wr, c])}"
                   for c in range(D_K)]))
        A("")
        A(table(H["Z"], toks, ["dim 0", "dim 1"]))
        A("")
    A(f"#### {sec}.{4 + N_HEADS} Concatenate and project\n")
    A(table(att["concat"], toks, HEAD_DIMS))
    A(f"\nWorked row **{toks[wr]}** of $`\\operatorname{{Concat}}(Z_1, Z_2)\\,W^O`$:\n")
    A(aligned([rf"o_{{{wr},{j}}} &= {dot(att['concat'][wr], ATTN['O'][:, j], db=1)} = {f(att['out'][wr, j])}"
               for j in range(D_MODEL)]))
    A("\n**Multi-head attention output**\n")
    A(table(att["out"], toks, DIMS))
    A("")


def doc_ffn(md, S, tokens, wr):
    A = md.append
    toks = labels(tokens)
    X1, F = S["ln1"]["out"], S["ffn"]
    A(math(r"\operatorname{FFN}(x) = \operatorname{GELU}(x W_1 + b_1)\, W_2 + b_2"))
    A("\nSame position-wise two-layer network as in the Transformer, but BERT uses **GELU** "
      "instead of ReLU. GELU is a smooth ReLU: $`\\text{GELU}(x) = x\\,\\Phi(x)`$ with "
      "$`\\Phi`$ the standard normal CDF; large positive inputs pass almost unchanged, large "
      "negative inputs give ≈ 0, and small negative inputs give small *negative* outputs. We use the "
      "tanh approximation from the original BERT code:\n")
    A(math(r"\operatorname{GELU}(x) \approx \tfrac12\, x \left(1 + \tanh\!\left[\sqrt{2/\pi}\,"
           r"\left(x + 0.044715\, x^3\right)\right]\right),\qquad \sqrt{2/\pi} \approx "
           + f(F["gelu"]["c"])))
    A("\n**Parameters**\n")
    A("$`W_1`$ (4×8)\n")
    A(table(FFN["W1"], W_ROWS, HIDDEN, d=1))
    A("\n$`b_1`$\n")
    A(table(FFN["b1"], ["b1"], HIDDEN, d=1))
    A("\n$`W_2`$ (8×4)\n")
    A(table(FFN["W2"], HIDDEN, DIMS, d=1))
    A("\n$`b_2`$\n")
    A(table(FFN["b2"], ["b2"], DIMS, d=1))
    A(f"\n**Step 1 – expand:** $`U = X_1W_1 + b_1`$. Worked row **{toks[wr]}**:\n")
    A(aligned([rf"u_{{{wr},{j}}} &= {dot(X1[wr], FFN['W1'][:, j], db=1)} + {fm(FFN['b1'][j], 1)} = {f(F['pre'][wr, j])}"
               for j in range(D_FF)]))
    A("")
    A(table(F["pre"], toks, HIDDEN))
    A(f"\n**Step 2 – GELU.** Worked row **{toks[wr]}**, step by step:\n")
    act = dict(F["gelu"], x=F["pre"])
    doc_gelu_row(md, act, HIDDEN, row=wr)
    A("\n$`\\operatorname{GELU}(U)`$ for all positions:\n")
    A(table(F["gelu"]["out"], toks, HIDDEN))
    A(f"\n**Step 3 – project back:** $`\\operatorname{{GELU}}(U)W_2 + b_2`$. Worked row **{toks[wr]}** "
      "(unlike ReLU, GELU leaves no exact zeros, so all 8 units contribute):\n")
    g = F["gelu"]["out"]
    A(aligned([rf"\text{{ffn}}_{{{wr},{c}}} &= {dot(g[wr], FFN['W2'][:, c], db=1)} + {fm(FFN['b2'][c], 1)} = {f(F['out'][wr, c])}"
               for c in range(D_MODEL)]))
    A("\n**FFN output**\n")
    A(table(F["out"], toks, DIMS))
    A("")


def doc_encoder(md, S, tokens, wr):
    toks = labels(tokens)
    md.append("## 2. The encoder layer\n")
    md.append("One BERT layer = multi-head self-attention → Add & Norm → feed-forward → Add & Norm "
              "(BERT-base stacks 12 such layers, each with its own weights).\n")
    md.append("### 2.1 Multi-head self-attention\n")
    doc_attention(md, "2.1", S, tokens, wr)
    md.append("### 2.2 Add & Norm (1)\n")
    doc_layer_norm(md, S["ln1"], toks, wr, "X_1", r"X + \operatorname{MultiHead}(X)",
                   S["emb_ln"]["out"], S["attn"]["out"])
    md.append("### 2.3 Feed-forward network with GELU\n")
    doc_ffn(md, S, tokens, wr)
    md.append("### 2.4 Add & Norm (2) → final hidden states $`H`$\n")
    doc_layer_norm(md, S["ln2"], toks, wr, "H", r"X_1 + \operatorname{FFN}(X_1)",
                   S["ln1"]["out"], S["ffn"]["out"])
    md.append("$`H`$ holds one contextual vector per input token — this is what “BERT embeddings” "
              "are. The pre-training head below reads from it.\n")


def mermaid(input_label, head_lines):
    return ("```mermaid\nflowchart TB\n"
            f"    a[\"{input_label}\"] --> b[\"1. token + segment + position embeddings, LayerNorm\"]\n"
            "    b --> c[\"2.1 Multi-head self-attention (2 heads, no mask)\"]\n"
            "    c --> d[\"2.2 Add & LayerNorm\"]\n"
            "    d --> e[\"2.3 Feed-forward 4 → 8 → 4 with GELU\"]\n"
            "    e --> f[\"2.4 Add & LayerNorm → H\"]\n"
            + head_lines + "```\n")


# --------------------------------------------------------------------------
# MLM.md
# --------------------------------------------------------------------------
def build_mlm(S):
    md = []
    A = md.append
    wr = MLM_MASK_POS
    toks = labels(MLM_TOKENS)
    hd = S["head"]
    target = VOCAB[hd["target"]]

    A("# BERT, pen and paper — Masked Language Modelling (MLM)\n")
    A("*A complete numerical walk-through of BERT's MLM pre-training task, from the raw sentence "
      "to the loss, for* **“socrates [MASK] plato”** *(original: “socrates taught plato”).*\n")
    A("> Every number is produced by [`bert_pen_and_paper.py`](bert_pen_and_paper.py); intermediate "
      "results are rounded to 4 decimals and the rounded value is used in the next step, so every "
      "entry can be checked with a calculator. The companion example is "
      "[NSP.md](NSP.md) (next sentence prediction) — same model, different head.\n")
    A("## Contents\n")
    A("0. [Setup](#0-setup) · 1. [Input representation](#1-input-representation) · "
      "2. [Encoder layer](#2-the-encoder-layer) · 3. [MLM head](#3-the-mlm-head) · "
      "4. [Loss and gradient](#4-loss-and-gradient) · [Summary](#summary) · [Exercises](#exercises)\n")
    A("## 0. Setup\n")
    task = (
        "**Task.** Hide some words of a sentence and train the model to recover them from the "
        "context on *both* sides. BERT selects 15 % of the (non-special) tokens; each selected "
        "token is replaced by `[MASK]` 80 % of the time, by a random token 10 % of the time, and "
        "left unchanged 10 % of the time (so that the model cannot rely on `[MASK]` being present, "
        "which it never is when BERT is later fine-tuned). The loss is computed **only at the "
        "selected positions**.\n\n"
        "Our sentence has three words, so 15 % rounds to one selected token: “taught”, and it "
        "falls in the 80 % `[MASK]` case. Predicting it needs the left context (“socrates”) and "
        "the right context (“plato”) — exactly what a left-to-right language model cannot do.\n\n"
        "| pos | 0 | 1 | 2 | 3 | 4 |\n|:--|:-:|:-:|:-:|:-:|:-:|\n"
        "| **original** | " + " | ".join(f"`{t}`" for t in MLM_ORIGINAL) + " |\n"
        "| **model input** | " + " | ".join(f"`{t}`" for t in MLM_TOKENS) + " |\n"
        "| **segment** | " + " | ".join("A" for _ in MLM_TOKENS) + " |\n"
        f"| **MLM label** | — | — | `{target}` | — | — |\n\n"
        "A single sentence uses segment A everywhere.\n")
    doc_setup(md, task)
    A("### 0.6 Pipeline\n")
    A(mermaid("[CLS] socrates [MASK] plato [SEP]",
              "    f -- \"row 2 ([MASK])\" --> g[\"3. MLM head: dense + GELU + LayerNorm\"]\n"
              "    g --> h[\"logits = t · E_tokᵀ + b_v, softmax over 8 words\"]\n"
              "    h --> i[\"4. loss = −log P(taught)\"]\n"))

    A("---\n\n## 1. Input representation\n")
    doc_input(md, S, MLM_TOKENS, MLM_SEGMENTS, wr, extra_rows=(1, 3))
    A("Note that the `[MASK]` row carries no information about the hidden word — only its "
      "position and segment. Everything the model knows about “taught” must flow in from the "
      "other tokens through attention.\n")

    A("---\n\n")
    doc_encoder(md, S, MLM_TOKENS, wr)
    A("Compare row 2 of $`H`$ with row 2 of $`X`$: after attention the `[MASK]` position "
      "has absorbed information from “socrates” and “plato”.\n")

    A("---\n\n## 3. The MLM head\n")
    A("Only the hidden state of the masked position, $`h = H_{2}`$, is needed (in practice the "
      "masked positions of a batch are gathered and the head runs on them only). The head is a "
      "small transform followed by a projection onto the vocabulary that is **tied to the token "
      "embedding matrix**:\n")
    A(math(r"\begin{aligned}"
           r"t &= \operatorname{LayerNorm}\big(\operatorname{GELU}(h\,W_t + b_t)\big) \\" "\n"
           r"\text{logits} &= t\,(E^{\text{tok}})^{\top} + b_v \qquad (8 \text{ scores, one per word}) \\" "\n"
           r"P &= \operatorname{softmax}(\text{logits})"
           r"\end{aligned}"))
    A("\n**Parameters**\n")
    A("$`W_t`$ (4×4)\n")
    A(table(MLM_HEAD["Wt"], W_ROWS, DIMS, d=1))
    A("\n$`b_t`$\n")
    A(table(MLM_HEAD["bt"], ["bt"], DIMS, d=1))
    A("\n$`b_v`$ (one bias per vocabulary word)\n")
    A(table(MLM_HEAD["bv"], ["bv"], VOCAB, d=1))
    A("\n### 3.1 Dense layer\n")
    A(f"$`h = H_2 = {vec(hd['h'])}`$\n")
    A(aligned([rf"(hW_t + b_t)_{{{j}}} &= {dot(hd['h'], MLM_HEAD['Wt'][:, j], db=1)} + {fm(MLM_HEAD['bt'][j], 1)} = {f(hd['pre'][j])}"
               for j in range(D_MODEL)]))
    A("\n### 3.2 GELU\n")
    doc_gelu_row(md, dict(hd["gelu"], x=hd["pre"]), DIMS)
    A("\n### 3.3 LayerNorm\n")
    doc_layer_norm(md, hd["ln"], ["t"], 0, "t", r"\operatorname{GELU}(hW_t + b_t)")
    t = hd["ln"]["out"][0]
    A("### 3.4 Scores for every vocabulary word\n")
    A("With weight tying, the score of word $`w`$ is the dot product of $`t`$ with that word's "
      "*input* embedding, plus its bias — words whose embedding points in the same direction as "
      "$`t`$ get high scores.\n")
    A(aligned([rf"\ell_{{\texttt{{{w}}}}} &= {dot(t, TOK[v], db=1)} + {fm(MLM_HEAD['bv'][v], 1)} = {f(hd['logits'][v])}"
               for v, w in enumerate(VOCAB)]))
    A("\n### 3.5 Softmax\n")
    A(aligned([r"&" + r",\quad ".join(rf"e^{{{f(hd['logits'][v])}}} = {f(hd['exps'][v])}" for v in range(k, k + 4))
               for k in (0, 4)]
              + [rf"&\textstyle\sum = " + " + ".join(f(x) for x in hd["exps"]) + f" = {f(hd['sum'])}"]))
    A("\n")
    A(table(np.vstack([hd["logits"], hd["exps"], hd["probs"]]),
            ["logit", "exp", "P (softmax)"], [f"`{w}`" for w in VOCAB]))
    pred = int(np.argmax(hd["probs"]))
    A(f"\nThe untrained model's top guess is **`{VOCAB[pred]}`** (P = {f(hd['probs'][pred])}); the "
      f"correct word `{target}` gets only P = {f(hd['probs'][hd['target']])}. "
      + ("(“socrates met plato” is a perfectly sensible sentence — but it is not the one in the "
         "training data.) " if VOCAB[pred] == "met" else "")
      + "Training will fix this.\n")

    A("## 4. Loss and gradient\n")
    A("Cross-entropy at the masked position (averaged over all masked positions when there are "
      "several; unmasked positions contribute nothing):\n")
    A(math(rf"\mathcal L_{{\text{{MLM}}}} = -\ln P(\texttt{{{target}}}) = -\ln {f(hd['probs'][hd['target']])} = {f(hd['loss'])}"))
    A(f"\nFor reference, uniform guessing over 8 words gives $`\\ln 8 = {f(np.log(8))}`$.\n")
    A("Back-propagation starts from the gradient of the loss with respect to the logits, "
      "$`P - \\text{onehot}(y)`$:\n")
    y = np.eye(len(VOCAB))[hd["target"]]
    A(math(rf"\frac{{\partial \mathcal L}}{{\partial\,\text{{logits}}}} = {vec(hd['probs'])} - {vec(y, 0)} = {vec(hd['probs'] - y)}"))
    A(f"\nOnly the entry of `{target}` is negative, so gradient descent raises its score and lowers "
      "all others. Because the output matrix is tied to $`E^{\\text{tok}}`$, this gradient also "
      f"moves the *input* embedding of `{target}` towards $`t`$; it then flows back through the "
      "head, the encoder layer and into the embeddings of “socrates” and “plato”, teaching them "
      "to carry information that predicts the missing verb.\n")

    A("---\n\n## Summary\n")
    A("| Step | Computation | Shape |\n|:--|:--|:-:|\n"
      "| Input | `[CLS] socrates [MASK] plato [SEP]` | 5 tokens |\n"
      "| Embeddings | $`E^{\\text{tok}} + E^{\\text{seg}} + E^{\\text{pos}}`$, LayerNorm | 5×4 |\n"
      "| Q, K, V | $`XW^Q,\\;XW^K,\\;XW^V`$ | 5×4 each |\n"
      "| Attention per head | $`\\operatorname{softmax}(Q_iK_i^{\\top}/\\sqrt2)\\,V_i`$ | 5×5 → 5×2 |\n"
      "| Multi-head output, Add & Norm | $`\\operatorname{LN}(X + [Z_1\\,Z_2]W^O)`$ | 5×4 |\n"
      "| FFN, Add & Norm | $`\\operatorname{LN}(X_1 + \\operatorname{GELU}(X_1W_1+b_1)W_2+b_2)`$ | 5×8 → 5×4 |\n"
      "| Masked hidden state | $`h = H_2`$ | 1×4 |\n"
      "| MLM transform | $`\\operatorname{LN}(\\operatorname{GELU}(hW_t+b_t))`$ | 1×4 |\n"
      "| Vocabulary scores | $`t\\,(E^{\\text{tok}})^{\\top} + b_v`$, softmax | 1×8 |\n"
      f"| Loss | $`-\\ln P(\\texttt{{{target}}})`$ = {f(hd['loss'])} | scalar |\n")
    A("## Exercises\n")
    A("1. Verify row 1 (“socrates”) of the head-1 attention weights by hand.\n"
      "2. Suppose the selected token had fallen in the 10 % “random token” case and been replaced "
      "by `wrote`. Which token-embedding row would be looked up at position 2? Is the MLM "
      "label different?\n"
      "3. Why does BERT not simply compute the loss at *every* position, like a language model? "
      "(Hint: what would the model see at an unmasked position?)\n"
      "4. Compute GELU and ReLU of $`-0.5`$, $`0`$ and $`2`$. Where do they differ most?\n"
      "5. Which logits in 3.4 are larger than that of `taught`? By how much would the output bias "
      "$`b_v`$ of `taught` have to grow for it to become the top prediction?\n"
      "6. The attention has no causal mask. Which entries of the head-1 weight matrix would be "
      "forced to zero in a left-to-right model, and why would that make this prediction harder?\n")
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# NSP.md
# --------------------------------------------------------------------------
def build_nsp(S):
    md = []
    A = md.append
    wr = 0
    toks = labels(NSP_TOKENS)
    hd = S["head"]
    seg = np.array(NSP_SEGMENTS)

    A("# BERT, pen and paper — Next Sentence Prediction (NSP)\n")
    A("*A complete numerical walk-through of BERT's NSP pre-training task, from the raw sentence "
      "pair to the loss, for* **A = “socrates taught plato”**, **B = “plato wrote”**.\n")
    A("> Every number is produced by [`bert_pen_and_paper.py`](bert_pen_and_paper.py); intermediate "
      "results are rounded to 4 decimals and the rounded value is used in the next step, so every "
      "entry can be checked with a calculator. The companion example is "
      "[MLM.md](MLM.md) (masked language modelling) — same model, different head.\n")
    A("## Contents\n")
    A("0. [Setup](#0-setup) · 1. [Input representation](#1-input-representation) · "
      "2. [Encoder layer](#2-the-encoder-layer) · 3. [NSP head](#3-the-nsp-head) · "
      "4. [Loss and gradient](#4-loss-and-gradient) · [Summary](#summary) · [Exercises](#exercises)\n")
    A("## 0. Setup\n")
    task = (
        "**Task.** Given two sentences A and B, decide whether B is the sentence that actually "
        "follows A in the corpus (**IsNext**) or a random sentence from elsewhere (**NotNext**). "
        "Training pairs are built 50 % / 50 %. The goal is to make BERT model relationships "
        "*between* sentences, which helps downstream tasks such as question answering and "
        "natural language inference.\n\n"
        "Both sentences are packed into one input, separated by `[SEP]`; **segment embeddings** "
        "tell the model which token belongs to which sentence; the decision is read from the "
        "`[CLS]` position.\n\n"
        "| pos | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |\n|:--|" + ":-:|" * 8 + "\n"
        "| **token** | " + " | ".join(f"`{t}`" for t in NSP_TOKENS) + " |\n"
        "| **segment** | " + " | ".join("AB"[s] for s in NSP_SEGMENTS) + " |\n\n"
        f"Label: **{NSP_CLASSES[NSP_LABEL]}** (B really follows A). A NotNext example would be, "
        "e.g., `[CLS] socrates taught plato [SEP] socrates met [SEP]` with B drawn at random. "
        "To isolate NSP we do not mask any token here; in real pre-training the same input "
        "would also carry MLM masks.\n")
    doc_setup(md, task)
    A("### 0.6 Pipeline\n")
    A(mermaid("[CLS] socrates taught plato [SEP] plato wrote [SEP]",
              "    f -- \"row 0 ([CLS])\" --> g[\"3. pooler: dense + tanh\"]\n"
              "    g --> h[\"classifier: 2 logits, softmax → P(IsNext), P(NotNext)\"]\n"
              "    h --> i[\"4. loss = −log P(IsNext)\"]\n"))

    A("---\n\n## 1. Input representation\n")
    doc_input(md, S, NSP_TOKENS, NSP_SEGMENTS, wr, extra_rows=(3, 5))
    A("Positions 3 and 5 are both `plato`: they share the token embedding but get different "
      "segment and position embeddings, so the model can tell the “plato” of sentence A from the "
      "“plato” of sentence B.\n")

    A("---\n\n")
    doc_encoder(md, S, NSP_TOKENS, wr)
    A("### 2.5 How much attention crosses the sentence boundary?\n")
    A("For NSP the model must compare the two sentences, so attention between segments matters. "
      "Share of each query's attention that lands on segment-A keys (positions 0–4) vs segment-B "
      "keys (positions 5–7):\n")
    rows = []
    for i, t in enumerate(toks):
        cells = []
        for H in S["attn"]["heads"]:
            a_share = H["A"][i, seg == 0].sum()
            cells += [f(a_share), f(H["A"][i, seg == 1].sum())]
        rows.append(f"| **{t}** | {'AB'[seg[i]]} | " + " | ".join(cells) + " |")
    A("| query | its segment | head 1 → A | head 1 → B | head 2 → A | head 2 → B |\n|:--|:-:|--:|--:|--:|--:|")
    A("\n".join(rows))
    cross = [np.mean([H["A"][i, seg != seg[i]].sum() for i in range(len(seg))]) for H in S["attn"]["heads"]]
    A(f"\nOn average a query sends {f(cross[0])} of its head-1 attention and {f(cross[1])} of its "
      "head-2 attention to the *other* sentence: the heads differ in how much they mix the two "
      "segments, which is possible only because the segment embeddings make the two sentences "
      "distinguishable. The `[CLS]` row — the only one the NSP head reads — collects information "
      "from both sentences.\n")

    A("---\n\n## 3. The NSP head\n")
    A("The head reads only the final hidden state of `[CLS]`, $`c = H_0`$. BERT first passes it "
      "through the **pooler** (a dense layer with tanh), then a 2-way linear classifier:\n")
    A(math(r"\begin{aligned}"
           r"p &= \tanh(c\,W_p + b_p) \qquad (\text{the “pooled output”}) \\" "\n"
           r"\text{logits} &= p\,W_c + b_c = [\ell_{\text{IsNext}},\; \ell_{\text{NotNext}}] \\" "\n"
           r"P &= \operatorname{softmax}(\text{logits})"
           r"\end{aligned}"))
    A("\nWhy `[CLS]`? It has no meaning of its own and sits at the same position in every input, "
      "so through attention it can become a summary of the whole pair. The pooled output $`p`$ is "
      "also what is used when BERT is later fine-tuned for sentence(-pair) classification.\n")
    A("**Parameters**\n")
    A("$`W_p`$ (4×4)\n")
    A(table(NSP_HEAD["Wp"], W_ROWS, DIMS, d=1))
    A("\n$`b_p`$\n")
    A(table(NSP_HEAD["bp"], ["bp"], DIMS, d=1))
    A("\n$`W_c`$ (4×2)\n")
    A(table(NSP_HEAD["Wc"], W_ROWS, NSP_CLASSES, d=1))
    A("\n$`b_c`$\n")
    A(table(NSP_HEAD["bc"], ["bc"], NSP_CLASSES, d=1))
    A("\n### 3.1 Pooler\n")
    A(f"$`c = H_0 = {vec(hd['c'])}`$\n")
    A(aligned([rf"(cW_p + b_p)_{{{j}}} &= {dot(hd['c'], NSP_HEAD['Wp'][:, j], db=1)} + {fm(NSP_HEAD['bp'][j], 1)} = {f(hd['pre'][j])}"
               for j in range(D_MODEL)]))
    A("")
    A(aligned([rf"p_{{{j}}} &= \tanh({f(hd['pre'][j])}) = {f(hd['pooled'][j])}" for j in range(D_MODEL)]))
    A("\n### 3.2 Classifier\n")
    A(aligned([rf"\ell_{{\text{{{name}}}}} &= {dot(hd['pooled'], NSP_HEAD['Wc'][:, k], db=1)} + {fm(NSP_HEAD['bc'][k], 1)} = {f(hd['logits'][k])}"
               for k, name in enumerate(NSP_CLASSES)]))
    A("\n### 3.3 Softmax\n")
    ex = hd["exps"]
    A(aligned([rf"&e^{{{f(hd['logits'][0])}}} = {f(ex[0])},\quad e^{{{f(hd['logits'][1])}}} = {f(ex[1])},"
               rf"\quad \textstyle\sum = {f(hd['sum'])}",
               rf"&P(\text{{IsNext}}) = {f(ex[0])} / {f(hd['sum'])} = {f(hd['probs'][0])},\qquad "
               rf"P(\text{{NotNext}}) = {f(ex[1])} / {f(hd['sum'])} = {f(hd['probs'][1])}"]))
    A("\nWith only two classes this is the same as a logistic sigmoid of the logit difference: "
      f"$`\\sigma(\\ell_{{\\text{{IsNext}}}} - \\ell_{{\\text{{NotNext}}}}) = "
      f"\\sigma({f(hd['logits'][0] - hd['logits'][1])}) \\approx {f(1 / (1 + np.exp(-(hd['logits'][0] - hd['logits'][1]))))}`$.\n")
    pred = int(np.argmax(hd["probs"]))
    A(f"The untrained model predicts **{NSP_CLASSES[pred]}** "
      f"(P = {f(hd['probs'][pred])}); the true label is **{NSP_CLASSES[NSP_LABEL]}**.\n")

    A("## 4. Loss and gradient\n")
    A(math(rf"\mathcal L_{{\text{{NSP}}}} = -\ln P(\text{{{NSP_CLASSES[NSP_LABEL]}}}) = "
           rf"-\ln {f(hd['probs'][NSP_LABEL])} = {f(hd['loss'])}"))
    A(f"\n(Guessing 50/50 would give $`\\ln 2 = {f(np.log(2))}`$.) The gradient with respect to the "
      "logits is again $`P - \\text{onehot}(y)`$:\n")
    y = np.eye(2)[NSP_LABEL]
    A(math(rf"\frac{{\partial \mathcal L}}{{\partial\,\text{{logits}}}} = {vec(hd['probs'])} - {vec(y, 0)} = {vec(hd['probs'] - y)}"))
    A("\nGradient descent raises $`\\ell_{\\text{IsNext}}`$ and lowers $`\\ell_{\\text{NotNext}}`$. "
      "Going backwards, the gradient passes through the classifier and the pooler (scaled by "
      "$`1 - p_j^2`$, the derivative of tanh) into $`H_0`$, and from there through attention into "
      "**every** token of both sentences — this is how the `[CLS]` vector learns to summarise the "
      "pair.\n")
    A("In real pre-training this loss is added to the MLM loss of the same input "
      "($`\\mathcal L = \\mathcal L_{\\text{MLM}} + \\mathcal L_{\\text{NSP}}`$). Later work "
      "(RoBERTa, 2019) found that dropping NSP and training on longer contiguous text works as "
      "well or better, and ALBERT replaced it with sentence-order prediction — a good discussion "
      "point on what the task really teaches.\n")

    A("---\n\n## Summary\n")
    A("| Step | Computation | Shape |\n|:--|:--|:-:|\n"
      "| Input | `[CLS] socrates taught plato [SEP] plato wrote [SEP]` | 8 tokens |\n"
      "| Embeddings | $`E^{\\text{tok}} + E^{\\text{seg}} + E^{\\text{pos}}`$, LayerNorm | 8×4 |\n"
      "| Q, K, V | $`XW^Q,\\;XW^K,\\;XW^V`$ | 8×4 each |\n"
      "| Attention per head | $`\\operatorname{softmax}(Q_iK_i^{\\top}/\\sqrt2)\\,V_i`$ | 8×8 → 8×2 |\n"
      "| Multi-head output, Add & Norm | $`\\operatorname{LN}(X + [Z_1\\,Z_2]W^O)`$ | 8×4 |\n"
      "| FFN, Add & Norm | $`\\operatorname{LN}(X_1 + \\operatorname{GELU}(X_1W_1+b_1)W_2+b_2)`$ | 8×8 → 8×4 |\n"
      "| `[CLS]` hidden state | $`c = H_0`$ | 1×4 |\n"
      "| Pooler | $`\\tanh(cW_p + b_p)`$ | 1×4 |\n"
      "| Classifier | $`pW_c + b_c`$, softmax | 1×2 |\n"
      f"| Loss | $`-\\ln P(\\text{{IsNext}})`$ = {f(hd['loss'])} | scalar |\n")
    A("## Exercises\n")
    A("1. Verify row 0 (`[CLS]`) of the head-2 attention weights by hand.\n"
      "2. Recompute the embedding sum of position 5 if `plato` were in segment A. Which later "
      "quantities would change?\n"
      "3. Build the NotNext input `[CLS] socrates taught plato [SEP] socrates met [SEP]`. Which "
      "rows of the token/segment/position lookup tables differ from this example?\n"
      "4. Show that for two classes, softmax cross-entropy equals binary cross-entropy with "
      "$`\\sigma(\\ell_{\\text{IsNext}} - \\ell_{\\text{NotNext}})`$.\n"
      "5. Compute $`\\partial \\mathcal L / \\partial p`$ (the gradient at the pooled output) and "
      "then $`\\partial \\mathcal L / \\partial (cW_p + b_p)`$ using $`\\tanh' = 1 - \\tanh^2`$.\n"
      "6. If a batch also contained a shorter pair padded with `[PAD]` tokens, how would you "
      "change the attention scores so that no token attends to padding?\n")
    return "\n".join(md) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare the rounded hand calculation with full precision")
    args = parser.parse_args()

    rnd = make_rounder(full_precision=False)
    mlm, nsp = mlm_forward(rnd), nsp_forward(rnd)
    here = Path(__file__).parent
    (here / "MLM.md").write_text(build_mlm(mlm), encoding="utf-8")
    (here / "NSP.md").write_text(build_nsp(nsp), encoding="utf-8")
    print(f"wrote {here / 'MLM.md'} and {here / 'NSP.md'}")

    if args.check:
        full = make_rounder(full_precision=True)
        np.set_printoptions(precision=6, suppress=True)
        for name, hand, ref in (("MLM", mlm, mlm_forward(full)), ("NSP", nsp, nsp_forward(full))):
            diff = np.abs(hand["head"]["probs"] - ref["head"]["probs"]).max()
            print(f"{name}: hand {hand['head']['probs']}\n     full {ref['head']['probs']}\n"
                  f"     max |difference| = {diff:.6f}, loss hand {hand['head']['loss']:.4f} "
                  f"vs full {ref['head']['loss']:.6f}")


if __name__ == "__main__":
    main()
