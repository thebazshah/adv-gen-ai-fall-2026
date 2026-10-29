#!/usr/bin/env python3
"""Pen-and-paper Transformer: generates README.md in this folder.

Every number in README.md is produced by this script, so the document and the
code can never disagree. Each intermediate result is rounded to 4 decimals and
the *rounded* value is what the next step uses, exactly as you would work with
a pocket calculator. Run

    python3 transformer_pen_and_paper.py          # regenerate README.md
    python3 transformer_pen_and_paper.py --check  # also compare to full precision

Only numpy is required.
"""

import argparse
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Hyperparameters, vocabulary and data
# --------------------------------------------------------------------------
VOCAB = ["Socrates", "was", "most", "prominent", "philosopher"]
D_MODEL, N_HEADS, D_FF = 4, 2, 8
D_K = D_MODEL // N_HEADS
DECIMALS = 4

ENC_IDS = [0, 1, 2, 3, 4]  # Socrates was most prominent philosopher
DEC_IDS = [0, 1, 2, 3]     # decoder input  = target shifted right
TARGET_IDS = [1, 2, 3, 4]  # decoder labels = next token at every position

# --------------------------------------------------------------------------
# Parameters (one decimal each so that hand multiplication stays easy)
# --------------------------------------------------------------------------
# Shared by encoder embedding, decoder embedding and the pre-softmax linear
# layer (weight tying, Vaswani et al. 2017, Sec. 3.4).
E = np.array([
    [0.9, -0.9, 0.2, 0.0],
    [-0.6, 0.0, 0.3, -0.5],
    [0.2, -0.8, 0.9, -1.0],
    [-0.2, -0.2, -0.9, -0.3],
    [0.3, -1.0, -1.0, -0.5],
])

ENC = {
    "attn": {
        "Q": np.array([
            [-0.8, 0.6, -0.8, 0.2],
            [0.7, 1.0, -0.8, 0.8],
            [1.0, 0.3, -0.3, -0.7],
            [-0.9, 0.7, 0.6, -0.2],
        ]),
        "K": np.array([
            [0.0, -1.0, -0.2, 0.9],
            [-0.1, 0.9, -0.2, -0.4],
            [0.5, 0.0, -0.3, -0.5],
            [0.3, 0.5, 0.5, 1.0],
        ]),
        "V": np.array([
            [-0.5, 0.2, -1.0, 0.2],
            [0.0, -0.6, -0.7, -0.5],
            [1.0, -0.2, -0.1, 0.1],
            [1.0, 0.5, 0.1, -1.0],
        ]),
        "O": np.array([
            [0.0, 0.1, -0.8, 1.0],
            [0.3, -0.6, -1.0, -0.2],
            [-0.9, -1.0, 0.4, 0.6],
            [0.6, -0.7, -1.0, 0.3],
        ]),
    },
    "W1": np.array([
        [-0.6, -0.6, -0.1, 0.9, 0.7, 0.8, 0.4, 0.1],
        [0.5, 0.3, 1.0, 0.7, -0.2, -0.7, 0.4, -0.7],
        [0.7, -1.0, 0.4, 0.0, -0.9, -0.1, -0.3, -0.6],
        [-0.9, 0.9, 0.2, 1.0, -1.0, 1.0, -0.5, 0.8],
    ]),
    "b1": np.array([0.5, -0.1, 0.1, -0.4, 0.7, 0.0, -0.2, -0.8]),
    "W2": np.array([
        [-0.8, -0.9, -0.6, -0.4],
        [0.6, 0.0, 1.0, -0.8],
        [0.8, 1.0, 0.5, -0.9],
        [-0.9, -0.5, 0.3, 0.5],
        [-0.6, -0.7, -1.0, -0.7],
        [0.9, -0.5, 0.2, -0.8],
        [1.0, 0.1, 0.2, 0.0],
        [-0.4, 0.3, -0.8, 0.9],
    ]),
    "b2": np.array([0.7, 0.2, 1.0, 0.2]),
}

DEC = {
    "self": {
        "Q": np.array([
            [0.1, 0.5, 0.6, -0.4],
            [1.0, -0.3, -0.9, 0.5],
            [1.0, -0.3, 0.3, -0.2],
            [0.1, -0.3, -0.2, 0.1],
        ]),
        "K": np.array([
            [-0.3, -0.4, -0.2, 0.9],
            [0.1, 0.1, 1.0, 0.0],
            [0.8, -0.8, -1.0, 0.2],
            [-0.8, 0.0, 0.7, 0.7],
        ]),
        "V": np.array([
            [-0.8, -0.6, 0.4, -0.7],
            [-0.9, -1.0, 0.4, 1.0],
            [-0.9, 0.7, -0.1, 0.8],
            [-0.1, -0.6, -0.5, -0.9],
        ]),
        "O": np.array([
            [-0.5, 1.0, 0.0, 0.7],
            [0.8, 0.5, -0.4, 0.1],
            [0.2, 0.3, -0.8, 0.6],
            [-0.6, 0.1, 1.0, 0.4],
        ]),
    },
    "cross": {
        "Q": np.array([
            [0.7, -0.6, -0.4, 0.1],
            [0.1, 0.4, 0.6, -0.7],
            [0.8, -0.4, 0.2, -1.0],
            [-0.2, 0.7, -0.8, -0.6],
        ]),
        "K": np.array([
            [-0.5, -0.1, 0.9, 0.5],
            [-0.2, -0.9, 0.7, -0.9],
            [0.3, 0.4, 0.3, 0.5],
            [-0.3, 0.1, -0.1, -1.0],
        ]),
        "V": np.array([
            [1.0, 0.0, 0.8, 1.0],
            [-0.6, 0.1, 1.0, -0.9],
            [0.3, 0.0, 0.3, 0.1],
            [0.4, 0.4, -0.6, 0.2],
        ]),
        "O": np.array([
            [-0.9, 0.2, 0.7, 0.2],
            [1.0, -0.5, -0.7, -0.3],
            [0.3, -0.4, 0.2, -0.5],
            [1.0, 0.2, -0.3, 0.9],
        ]),
    },
    "W1": np.array([
        [0.5, 0.1, 0.8, -0.5, 0.7, 0.3, 0.7, 0.1],
        [0.8, -1.0, -0.3, -0.6, -0.1, -0.8, 0.0, 0.6],
        [0.8, -0.4, -0.1, 0.9, 0.2, -0.5, 0.1, 0.0],
        [-1.0, -0.1, -0.8, 0.4, 0.8, -0.5, -0.9, -0.5],
    ]),
    "b1": np.array([1.0, 0.6, -0.9, -0.6, 0.9, 0.5, -1.0, -0.5]),
    "W2": np.array([
        [0.7, -1.0, -0.2, -0.3],
        [0.1, -0.5, 0.6, -0.1],
        [-1.0, -0.6, 0.4, 0.2],
        [-0.8, 0.1, -0.4, 0.6],
        [-0.8, 0.4, -0.2, 0.4],
        [0.3, 0.3, 0.8, -0.6],
        [0.1, 0.9, 0.2, 0.1],
        [0.1, -0.4, -0.7, 0.6],
    ]),
    "b2": np.array([0.9, 0.2, 0.4, 0.3]),
}


# --------------------------------------------------------------------------
# The model. `rnd` is applied after every step: rounding for the hand-
# calculation version, identity for the full-precision reference.
# --------------------------------------------------------------------------
def make_rounder(full_precision):
    if full_precision:
        return lambda x: np.asarray(x, dtype=float)
    return lambda x: np.round(np.asarray(x, dtype=float), DECIMALS) + 0.0  # +0.0 drops "-0"


def positional_encoding(n, rnd):
    angles = np.array([[pos / 10000 ** (2 * i / D_MODEL) for i in range(D_MODEL // 2)]
                       for pos in range(n)])
    pe = np.zeros((n, D_MODEL))
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return angles, rnd(pe)


def softmax(x, rnd):
    exps = rnd(np.exp(x))
    sums = rnd(exps.sum(axis=-1))
    return exps, sums, rnd(exps / sums[:, None])


def layer_norm(x, rnd):
    mu = rnd(x.mean(axis=1))
    var = rnd(((x - mu[:, None]) ** 2).mean(axis=1))
    sd = rnd(np.sqrt(var))
    return {"inp": x, "mu": mu, "var": var, "sd": sd, "out": rnd((x - mu[:, None]) / sd[:, None])}


def multi_head_attention(xq, xkv, W, rnd, causal=False):
    Q, K, V = rnd(xq @ W["Q"]), rnd(xkv @ W["K"]), rnd(xkv @ W["V"])
    scale = rnd(np.sqrt(D_K))
    heads = []
    for h in range(N_HEADS):
        cols = slice(h * D_K, (h + 1) * D_K)
        q, k, v = Q[:, cols], K[:, cols], V[:, cols]
        scores = rnd(q @ k.T)
        scaled = rnd(scores / scale)
        if causal:
            future = np.triu(np.ones(scaled.shape, dtype=bool), k=1)
            scaled = np.where(future, -np.inf, scaled)
        exps, sums, A = softmax(scaled, rnd)
        heads.append({"q": q, "k": k, "v": v, "scores": scores, "scaled": scaled,
                      "exps": exps, "sums": sums, "A": A, "Z": rnd(A @ v)})
    concat = np.hstack([hd["Z"] for hd in heads])
    return {"Q": Q, "K": K, "V": V, "scale": scale, "heads": heads,
            "concat": concat, "out": rnd(concat @ W["O"])}


def feed_forward(x, P, rnd):
    pre = rnd(x @ P["W1"] + P["b1"])
    hidden = np.maximum(pre, 0.0)
    return {"pre": pre, "hidden": hidden, "out": rnd(hidden @ P["W2"] + P["b2"])}


def embed(ids, rnd):
    emb = E[ids]
    scaled = rnd(emb * np.sqrt(D_MODEL))
    angles, pe = positional_encoding(len(ids), rnd)
    return {"emb": emb, "scaled": scaled, "angles": angles, "pe": pe, "x": rnd(scaled + pe)}


def forward(rnd, dec_ids=DEC_IDS):
    S = {}
    # ---- encoder
    S["enc_in"] = embed(ENC_IDS, rnd)
    X = S["enc_in"]["x"]
    S["enc_attn"] = multi_head_attention(X, X, ENC["attn"], rnd)
    S["enc_ln1"] = layer_norm(rnd(X + S["enc_attn"]["out"]), rnd)
    X1 = S["enc_ln1"]["out"]
    S["enc_ffn"] = feed_forward(X1, ENC, rnd)
    S["enc_ln2"] = layer_norm(rnd(X1 + S["enc_ffn"]["out"]), rnd)
    M = S["enc_ln2"]["out"]
    # ---- decoder
    S["dec_in"] = embed(dec_ids, rnd)
    Y = S["dec_in"]["x"]
    S["dec_self"] = multi_head_attention(Y, Y, DEC["self"], rnd, causal=True)
    S["dec_ln1"] = layer_norm(rnd(Y + S["dec_self"]["out"]), rnd)
    Y1 = S["dec_ln1"]["out"]
    S["dec_cross"] = multi_head_attention(Y1, M, DEC["cross"], rnd)
    S["dec_ln2"] = layer_norm(rnd(Y1 + S["dec_cross"]["out"]), rnd)
    Y2 = S["dec_ln2"]["out"]
    S["dec_ffn"] = feed_forward(Y2, DEC, rnd)
    S["dec_ln3"] = layer_norm(rnd(Y2 + S["dec_ffn"]["out"]), rnd)
    Y3 = S["dec_ln3"]["out"]
    # ---- output layer, loss
    S["logits"] = rnd(Y3 @ E.T)
    S["out_exps"], S["out_sums"], S["probs"] = softmax(S["logits"], rnd)
    n = len(dec_ids)
    S["losses"] = rnd(-np.log(S["probs"][np.arange(n), TARGET_IDS[:n]]))
    S["loss"] = rnd(S["losses"].mean())
    return S


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------
def f(x, d=DECIMALS):
    """Number as text; -inf becomes the minus-infinity sign (for tables)."""
    if np.isneginf(x):
        return "−∞"
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def fm(x, d=DECIMALS):
    """Number for LaTeX: negatives in parentheses so products read clearly."""
    if np.isneginf(x):
        return r"(-\infty)"
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


def tok(t):
    return rf"\text{{{t}}}"


DIMS = [f"dim {j}" for j in range(D_MODEL)]
HEAD_DIMS = [f"dim {j} (head {j // D_K + 1})" for j in range(D_MODEL)]
HIDDEN = [f"u{j}" for j in range(D_FF)]
W_ROWS = [f"in {j}" for j in range(D_MODEL)]
ENC_TOK = [VOCAB[i] for i in ENC_IDS]
DEC_TOK = [VOCAB[i] for i in DEC_IDS]


# --------------------------------------------------------------------------
# Reusable document sections
# --------------------------------------------------------------------------
def doc_weights(md, W, names):
    for key, label in names:
        md.append(f"$`{label}`$ (4×4)\n")
        md.append(table(W[key], W_ROWS, DIMS if key == "O" else HEAD_DIMS, d=1))
        md.append("")


def doc_attention(md, sec, A, W, xq, xkv, q_name, kv_name, q_tok, k_tok, wr, causal=False):
    """Q/K/V projections, every head, concatenation and output projection."""
    md.append(f"#### {sec}.2 Project to queries, keys and values\n")
    md.append(math(rf"Q = {q_name}\,W^Q,\qquad K = {kv_name}\,W^K,\qquad V = {kv_name}\,W^V"))
    md.append(f"\nShapes: $`Q`$ is {len(q_tok)}×4, $`K`$ and $`V`$ are {len(k_tok)}×4. "
              f"Worked example: row **{q_tok[wr]}** of $`Q`$ and row **{k_tok[0]}** of "
              f"$`K`$ and $`V`$ (row of the input times each column of the weight matrix):\n")
    lines = []
    for name, src, r, mat in (("q", xq, wr, "Q"), ("k", xkv, 0, "K"), ("v", xkv, 0, "V")):
        t = q_tok[wr] if name == "q" else k_tok[0]
        for j in range(D_MODEL):
            lines.append(rf"{name}_{{{tok(t)},{j}}} &= {dot(src[r], W[mat][:, j], db=1)} = {f(A[mat][r, j])}")
    md.append(aligned(lines))
    for mat, toks in (("Q", q_tok), ("K", k_tok), ("V", k_tok)):
        md.append(f"\n**{mat}**\n")
        md.append(table(A[mat], toks, HEAD_DIMS))
    md.append("")

    md.append(f"#### {sec}.3 Split into {N_HEADS} heads\n")
    md.append("Multi-head attention does not need separate weight matrices per head: "
              f"head 1 simply uses columns 0–1 of $`Q, K, V`$ and head 2 uses columns 2–3 "
              f"(i.e. $`W^Q_1`$ = columns 0–1 of $`W^Q`$, etc.). Each head works in a "
              f"$`d_k = d_{{\\text{{model}}}}/h = {D_K}`$-dimensional subspace.\n")
    md.append(math(r"\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}"
                   + (r" + \text{Mask}" if causal else "") + r"\right) V_i,"
                   rf"\qquad \sqrt{{d_k}} = \sqrt{{{D_K}}} \approx {f(A['scale'])}"))
    md.append("")

    for h, H in enumerate(A["heads"], start=1):
        hs = f"{sec}.{3 + h}"
        md.append(f"#### {hs} Head {h}\n")
        md.append(f"**Inputs of head {h}** — $`Q_{h}`$, $`K_{h}`$, $`V_{h}`$ (columns "
                  f"{2 * h - 2}–{2 * h - 1} of $`Q, K, V`$):\n")
        if q_tok == k_tok:
            qkv = np.hstack([H["q"], H["k"], H["v"]])
            md.append(table(qkv, q_tok, [f"{m}{h}[{c}]" for m in "QKV" for c in range(D_K)]))
        else:
            md.append(f"Queries (one per decoder position):\n")
            md.append(table(H["q"], q_tok, [f"Q{h}[{c}]" for c in range(D_K)]))
            md.append(f"\nKeys and values (one per encoder position):\n")
            md.append(table(np.hstack([H["k"], H["v"]]), k_tok,
                            [f"{m}{h}[{c}]" for m in "KV" for c in range(D_K)]))
        md.append(f"\n**Step 1 – raw scores** $`S_{h} = Q_{h} K_{h}^{{\\top}}`$ "
                  f"({len(q_tok)}×{len(k_tok)}). Entry $`(i, j)`$ is the dot product of query $`i`$ "
                  f"with key $`j`$. Worked row **{q_tok[wr]}**:\n")
        lines = [rf"s_{{{tok(q_tok[wr])},{tok(k_tok[j])}}} &= {dot(H['q'][wr], H['k'][j])} = {f(H['scores'][wr, j])}"
                 for j in range(len(k_tok))]
        md.append(aligned(lines))
        md.append("")
        md.append(table(H["scores"], q_tok, k_tok))

        md.append(f"\n**Step 2 – scale** by $`\\sqrt{{d_k}} = {f(A['scale'])}`$ "
                  f"(e.g. $`{fm(H['scores'][wr, 0])} / {f(A['scale'])} = {f(H['scores'][wr, 0] / A['scale'])}`$"
                  + (")" if not causal else "), then **apply the causal mask** (add $`-\\infty`$ "
                                                "wherever the key is in the future, $`j > i`$)") + ":\n")
        md.append(table(H["scaled"], q_tok, k_tok))

        n_vis = len(k_tok) if not causal else wr + 1
        md.append(f"\n**Step 3 – softmax over each row.** Worked row **{q_tok[wr]}**"
                  + (f" (only the first {n_vis} keys are visible; $`e^{{-\\infty}} = 0`$)" if causal else "")
                  + ":\n")
        powers = [r"-\infty" if np.isneginf(x) else f(x) for x in H["scaled"][wr]]
        e_terms = [rf"e^{{{p}}} = {f(H['exps'][wr, j])}" for j, p in enumerate(powers)]
        lines = [r"&" + r",\quad ".join(e_terms[:3]) + ",",
                 r"&" + r",\quad ".join(e_terms[3:]) if len(e_terms) > 3 else None,
                 rf"&\textstyle\sum = " + " + ".join(f(x) for x in H["exps"][wr]) + f" = {f(H['sums'][wr])}",
                 r"&a_{" + tok(q_tok[wr]) + r",\cdot} = \frac{1}{" + f(H["sums"][wr]) + r"}"
                 + vec(H["exps"][wr]) + " = " + vec(H["A"][wr])]
        md.append(aligned([l for l in lines if l]))
        md.append(f"\n**Attention weights** $`A_{h}`$ (each row sums to 1, up to rounding; "
                  "row = query, column = key):\n")
        md.append(table(H["A"], q_tok, k_tok))
        best = [f"{q_tok[i]} → **{k_tok[int(np.argmax(H['A'][i]))]}** ({f(H['A'][i].max())})"
                for i in range(len(q_tok))]
        md.append(f"\nStrongest link per query: " + "; ".join(best) + ".")

        md.append(f"\n**Step 4 – weighted sum of values** $`Z_{h} = A_{h} V_{h}`$ "
                  f"({len(q_tok)}×{D_K}). Worked row **{q_tok[wr]}**:\n")
        lines = [rf"z_{{{tok(q_tok[wr])},{c}}} &= {dot(H['A'][wr], H['v'][:, c])} = {f(H['Z'][wr, c])}"
                 for c in range(D_K)]
        md.append(aligned(lines))
        md.append("")
        md.append(table(H["Z"], q_tok, ["dim 0", "dim 1"]))
        md.append("")

    hs = f"{sec}.{4 + N_HEADS}"
    md.append(f"#### {hs} Concatenate heads and apply the output projection\n")
    md.append(math(r"\operatorname{MultiHead} = \operatorname{Concat}(\text{head}_1, \text{head}_2)\, W^O"))
    md.append("\n**Concat**$`(Z_1, Z_2)`$ (columns 0–1 from head 1, 2–3 from head 2):\n")
    md.append(table(A["concat"], q_tok, HEAD_DIMS))
    md.append(f"\nWorked row **{q_tok[wr]}**:\n")
    lines = [rf"o_{{{tok(q_tok[wr])},{j}}} &= {dot(A['concat'][wr], W['O'][:, j], db=1)} = {f(A['out'][wr, j])}"
             for j in range(D_MODEL)]
    md.append(aligned(lines))
    md.append("\n**Multi-head attention output**\n")
    md.append(table(A["out"], q_tok, DIMS))
    md.append("")


def doc_add_norm(md, ln, a_name, b_name, a_mat, b_mat, out_name, toks, wr):
    md.append(math(rf"{out_name} = \operatorname{{LayerNorm}}({a_name} + {b_name}),\qquad "
                   r"\operatorname{LayerNorm}(r) = \gamma \odot \frac{r - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta"))
    md.append("\nwith $`\\mu`$ and $`\\sigma^2`$ the mean and (population) variance of the 4 "
              "entries **of each row** (each token is normalised on its own), "
              "$`\\gamma = \\mathbf 1`$, $`\\beta = \\mathbf 0`$, $`\\epsilon \\approx 0`$.\n")
    md.append("**Residual sum** (element-wise):\n")
    md.append(table(ln["inp"], toks, DIMS))
    r, mu, var, sd = ln["inp"][wr], ln["mu"][wr], ln["var"][wr], ln["sd"][wr]
    md.append(f"\nWorked row **{toks[wr]}**:\n")
    lines = [
        rf"r &= {vec(a_mat[wr])} + {vec(b_mat[wr])} = {vec(r)}",
        rf"\mu &= \tfrac14({' + '.join(fm(x) for x in r)}) = {f(mu)}",
        rf"\sigma^2 &= \tfrac14\left({' + '.join(f'{fm(x - mu)}^2' for x in r)}\right) = {f(var)}",
        rf"\sigma &= \sqrt{{{f(var)}}} = {f(sd)}",
        rf"\hat r &= \frac{{r - {fm(mu)}}}{{{f(sd)}}} = {vec(ln['out'][wr])}",
    ]
    md.append(aligned(lines))
    md.append(f"\n| | μ | σ² | σ |\n|:--|--:|--:|--:|")
    md.append("\n".join(f"| **{t}** | {f(ln['mu'][i])} | {f(ln['var'][i])} | {f(ln['sd'][i])} |"
                        for i, t in enumerate(toks)))
    md.append(f"\n$`{out_name}`$ (every row now has mean 0 and variance 1):\n")
    md.append(table(ln["out"], toks, DIMS))
    md.append("")


def doc_ffn(md, F, P, in_name, in_mat, toks, wr):
    md.append(math(r"\operatorname{FFN}(x) = \max(0,\; x W_1 + b_1)\, W_2 + b_2,"
                   rf"\qquad W_1 \in \mathbb R^{{{D_MODEL}\times{D_FF}}},\; W_2 \in \mathbb R^{{{D_FF}\times{D_MODEL}}}"))
    md.append("\nThe same two-layer network is applied to every position separately "
              "(“position-wise”); it expands to $`d_{ff} = 8`$ and projects back to 4.\n")
    md.append("**Parameters**\n")
    md.append(f"$`W_1`$ (4×8)\n")
    md.append(table(P["W1"], W_ROWS, HIDDEN, d=1))
    md.append(f"\n$`b_1`$\n")
    md.append(table(P["b1"], ["b1"], HIDDEN, d=1))
    md.append(f"\n$`W_2`$ (8×4)\n")
    md.append(table(P["W2"], [f"u{j}" for j in range(D_FF)], DIMS, d=1))
    md.append(f"\n$`b_2`$\n")
    md.append(table(P["b2"], ["b2"], DIMS, d=1))
    md.append(f"\n**Step 1 – expand:** $`U = {in_name} W_1 + b_1`$. Worked row **{toks[wr]}**:\n")
    lines = [rf"u_{{{j}}} &= {dot(in_mat[wr], P['W1'][:, j], db=1)} + {fm(P['b1'][j], 1)} = {f(F['pre'][wr, j])}"
             for j in range(D_FF)]
    md.append(aligned(lines))
    md.append("")
    md.append(table(F["pre"], toks, HIDDEN))
    md.append(f"\n**Step 2 – ReLU:** negative entries become 0.\n")
    md.append(table(F["hidden"], toks, HIDDEN))
    active = [j for j in range(D_FF) if F["hidden"][wr, j] > 0]
    md.append(f"\n**Step 3 – project back:** $`\\operatorname{{FFN}} = \\operatorname{{ReLU}}(U) W_2 + b_2`$. "
              f"Worked row **{toks[wr]}** — only the active units "
              f"({', '.join(f'u{j}' for j in active)}) contribute, the zeros drop out:\n")
    lines = [rf"\text{{ffn}}_{{{c}}} &= " + " + ".join(f"{fm(F['hidden'][wr, j])}\\cdot {fm(P['W2'][j, c], 1)}"
                                                       for j in active)
             + f" + {fm(P['b2'][c], 1)} = {f(F['out'][wr, c])}" for c in range(D_MODEL)]
    md.append(aligned(lines))
    md.append("\n**FFN output**\n")
    md.append(table(F["out"], toks, DIMS))
    md.append("")


def doc_embedding(md, emb, toks, wr, who):
    md.append(f"Each token id selects one row of the shared embedding matrix $`E`$ "
              f"(equivalently: one-hot vector × $`E`$). For **{toks[wr]}** (id {VOCAB.index(toks[wr])}):\n")
    one_hot = [1 if i == VOCAB.index(toks[wr]) else 0 for i in range(len(VOCAB))]
    md.append(math(rf"e_{{{tok(toks[wr])}}} = \underbrace{{{vec(one_hot, 0)}}}_{{\text{{one-hot}}}}\, E = "
                   rf"{vec(emb['emb'][wr], 1)}\quad(\text{{row {VOCAB.index(toks[wr])} of }} E)"))
    md.append(f"\nThe paper multiplies embeddings by $`\\sqrt{{d_{{\\text{{model}}}}}} = \\sqrt 4 = 2`$ "
              f"so that they are not drowned out by the positional encodings. "
              f"{who} embeddings $`\\times 2`$:\n")
    md.append(table(emb["scaled"], toks, DIMS))
    md.append("")


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------
def build_markdown(S, S_first_step):
    md = []
    A = md.append
    enc_wr = 0      # worked row in the encoder: "Socrates"
    self_wr = 2     # worked row in masked self-attention: "most" (sees 3 tokens)
    cross_wr = 3    # worked row in cross-attention / output: "prominent"

    A("# A Pen-and-Paper Transformer\n")
    A("*A complete numerical walk-through of the encoder–decoder Transformer of "
      "Vaswani et al., “Attention Is All You Need” (2017), with a vocabulary of only "
      "five tokens and the sentence* **“Socrates was most prominent philosopher”.**\n")
    A("> Every number in this document is produced by "
      "[`transformer_pen_and_paper.py`](transformer_pen_and_paper.py). Each intermediate "
      "result is rounded to **4 decimals** and the rounded value is what the next step uses, "
      "so you can reproduce every entry with a calculator. "
      "(Run `python3 transformer_pen_and_paper.py --check` to see that rounding changes the "
      "final probabilities only in the 4th decimal.)\n")

    A("## Contents\n")
    A("0. [Setup](#0-setup)\n"
      "1. **Encoder** — [A.1 Embeddings](#a1-token-embeddings) · "
      "[A.2 Positional encoding](#a2-positional-encoding) · "
      "[A.3 Encoder input](#a3-encoder-input-x--embedding--positional-encoding) · "
      "[A.4 Multi-head self-attention](#a4-multi-head-self-attention) · "
      "[A.5 Add & Norm](#a5-add--norm-1) · [A.6 Feed-forward](#a6-position-wise-feed-forward-network) · "
      "[A.7 Add & Norm → encoder output](#a7-add--norm-2--encoder-output-m)\n"
      "2. **Decoder** — [B.1 Decoder input](#b1-decoder-input-shifted-right-embeddings-and-positions) · "
      "[B.2 Masked self-attention](#b2-masked-multi-head-self-attention) · "
      "[B.3 Add & Norm](#b3-add--norm-1) · [B.4 Cross-attention](#b4-encoderdecoder-cross-attention) · "
      "[B.5 Add & Norm](#b5-add--norm-2) · [B.6 Feed-forward](#b6-position-wise-feed-forward-network) · "
      "[B.7 Add & Norm](#b7-add--norm-3--decoder-output) · "
      "[B.8 Linear + softmax](#b8-final-linear-layer-and-softmax) · [B.9 Loss](#b9-training-loss-cross-entropy) · "
      "[B.10 Inference](#b10-inference-autoregressive-generation)\n"
      "3. [Summary of shapes](#summary-of-shapes) · [Exercises](#exercises)\n")

    # ------------------------------------------------------------------ setup
    A("## 0. Setup\n")
    A("### 0.1 Vocabulary\n")
    A("Word-level tokenisation, $`|V| = 5`$:\n")
    A("| Token id | 0 | 1 | 2 | 3 | 4 |\n|:--|:-:|:-:|:-:|:-:|:-:|\n| **Token** | "
      + " | ".join(VOCAB) + " |\n")
    A("### 0.2 Hyperparameters\n")
    A("| Symbol | Meaning | Paper (base model) | This example |\n|:--|:--|--:|--:|\n"
      "| $`d_{\\text{model}}`$ | width of every token vector | 512 | **4** |\n"
      "| $`h`$ | number of attention heads | 8 | **2** |\n"
      "| $`d_k = d_v = d_{\\text{model}}/h`$ | width of one head | 64 | **2** |\n"
      "| $`d_{ff}`$ | hidden width of the feed-forward network | 2048 | **8** |\n"
      "| $`N`$ | layers in encoder and in decoder | 6 | **1** |\n"
      "| $`\\lvert V\\rvert`$ | vocabulary size | 37 000 | **5** |\n")
    A("Further simplifications (none of them changes the mechanics): no dropout; the Q/K/V/O "
      "projections have no bias; LayerNorm uses $`\\gamma = 1, \\beta = 0`$ and "
      "$`\\epsilon \\approx 0`$. As in the paper (§3.4) one embedding matrix $`E`$ is shared by the "
      "encoder input, the decoder input and the final pre-softmax linear layer. All weights are "
      "made-up “untrained” values with one decimal so that multiplications stay easy.\n")
    A("### 0.3 The task\n")
    A("A Transformer is a sequence-to-sequence model: the **encoder** reads a source sentence and "
      "the **decoder** generates a target sentence one token at a time. With only five tokens there "
      "is no room for special tokens (`<bos>`, `<eos>`, `<pad>`), so we use a toy task: the encoder "
      "reads the whole sentence and the decoder learns to (re)generate it. During training the "
      "decoder sees the target *shifted right by one* (teacher forcing) and must predict the next "
      "token at every position; the first word “Socrates” plays the role of the start token.\n")
    A("| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 |\n|:--|:-:|:-:|:-:|:-:|:-:|\n"
      "| **Encoder input** | " + " | ".join(ENC_TOK) + " |\n"
      "| **Decoder input** | " + " | ".join(DEC_TOK) + " | — |\n"
      "| **Decoder target** | " + " | ".join(VOCAB[i] for i in TARGET_IDS) + " | — |\n")
    A("In a real system (e.g. translation) the decoder input would be `<bos> Socrates … philosopher` "
      "and the target `Socrates … philosopher <eos>`; every computation below stays exactly the same.\n")
    A("### 0.4 Conventions\n")
    A("* **Row vectors.** Each token is a row; a sentence of $`n`$ tokens is an $`n \\times d`$ "
      "matrix, and a linear layer is $`XW`$ (as in the paper's formulas).\n"
      "* Positions, dimensions and token ids are counted from 0; heads from 1.\n"
      "* $`\\operatorname{softmax}(z)_j = e^{z_j} / \\sum_k e^{z_k}`$, applied to each row separately.\n"
      "* Weight-matrix tables: rows = input dimension (“in 0…3”), columns = output dimension.\n")
    A("### 0.5 Architecture at a glance\n")
    A("```mermaid\nflowchart TB\n"
      "    subgraph ENC[\"Encoder (N = 1)\"]\n        direction TB\n"
      "        e1[\"Socrates was most prominent philosopher\"] --> e2[\"A.1–A.3 Embedding × 2 + positional encoding\"]\n"
      "        e2 --> e3[\"A.4 Multi-head self-attention (2 heads)\"]\n"
      "        e3 --> e4[\"A.5 Add & LayerNorm\"]\n"
      "        e4 --> e5[\"A.6 Feed-forward 4 → 8 → 4\"]\n"
      "        e5 --> e6[\"A.7 Add & LayerNorm\"]\n    end\n"
      "    subgraph DEC[\"Decoder (N = 1)\"]\n        direction TB\n"
      "        d1[\"Socrates was most prominent (shifted right)\"] --> d2[\"B.1 Embedding × 2 + positional encoding\"]\n"
      "        d2 --> d3[\"B.2 Masked multi-head self-attention\"]\n"
      "        d3 --> d4[\"B.3 Add & LayerNorm\"]\n"
      "        d4 --> d5[\"B.4 Cross-attention: Q from decoder, K and V from encoder\"]\n"
      "        d5 --> d6[\"B.5 Add & LayerNorm\"]\n"
      "        d6 --> d7[\"B.6 Feed-forward 4 → 8 → 4\"]\n"
      "        d7 --> d8[\"B.7 Add & LayerNorm\"]\n"
      "        d8 --> d9[\"B.8 Linear (Eᵀ) + softmax → P(next token)\"]\n    end\n"
      "    e6 -- \"memory M (5×4)\" --> d5\n```\n")
    A("Every “Add & LayerNorm” adds the sub-layer's input back to its output (a residual "
      "connection) before normalising.\n")
    A("### 0.6 Shared embedding matrix $`E`$ (5×4)\n")
    A(table(E, [f"{i}: {t}" for i, t in enumerate(VOCAB)], DIMS, d=1))
    A("")

    # ---------------------------------------------------------------- encoder
    A("---\n\n# Part A — Encoder\n")
    A("### A.1 Token embeddings\n")
    doc_embedding(md, S["enc_in"], ENC_TOK, enc_wr, "Scaled")

    angles, pe = S["enc_in"]["angles"], S["enc_in"]["pe"]
    A("### A.2 Positional encoding\n")
    A("Self-attention treats its input as a *set*, so word order must be injected explicitly. "
      "The paper adds a fixed sinusoidal vector to each position:\n")
    A(math(r"PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad "
           r"PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)"))
    A("\nWith $`d_{\\text{model}} = 4`$ there are two frequency pairs:\n")
    A(math(r"\begin{aligned}"
           r"i = 0 &: \; 10000^{0/4} = 1 &&\Rightarrow (\text{dim }0, \text{dim }1) = (\sin(pos),\; \cos(pos)) \\" "\n"
           r"i = 1 &: \; 10000^{2/4} = 100 &&\Rightarrow (\text{dim }2, \text{dim }3) = "
           r"\left(\sin\tfrac{pos}{100},\; \cos\tfrac{pos}{100}\right)"
           r"\end{aligned}"))
    A("\nAngles are in **radians**. Dimensions 0–1 oscillate quickly (period $`2\\pi`$ ≈ 6.3 positions), "
      "dimensions 2–3 very slowly (period ≈ 628 positions).\n")
    A("| pos | token | angle $`pos/1`$ | angle $`pos/100`$ |\n|:-:|:--|--:|--:|")
    A("\n".join(f"| {p} | {ENC_TOK[p]} | {angles[p, 0]:.2f} | {angles[p, 1]:.2f} |" for p in range(len(ENC_IDS))))
    A(f"\nWorked example, **pos 2 (“{ENC_TOK[2]}”)**:\n")
    A(math(rf"PE_2 = \left[\sin 2,\; \cos 2,\; \sin 0.02,\; \cos 0.02\right] = {vec(pe[2])}"))
    A("\n**Positional encodings**\n")
    A(table(pe, [f"pos {p}" for p in range(len(ENC_IDS))], DIMS))
    A("\nBecause $`\\sin(a+b)`$ and $`\\cos(a+b)`$ are linear combinations of $`\\sin a`$ and "
      "$`\\cos a`$, the encoding of position $`pos + k`$ is a fixed linear function (a rotation) of "
      "the encoding of $`pos`$; this makes relative positions easy for attention to pick up.\n")

    X = S["enc_in"]["x"]
    A("### A.3 Encoder input $`X`$ = embedding + positional encoding\n")
    A(math(r"X = \sqrt{d_{\text{model}}}\; E[\text{tokens}] + PE"))
    A(f"\nWorked row **{ENC_TOK[2]}**: $`{vec(S['enc_in']['scaled'][2])} + {vec(pe[2])} = {vec(X[2])}`$\n")
    A("**X** (5×4) — the input to the encoder layer:\n")
    A(table(X, ENC_TOK, DIMS))
    A("")

    A("### A.4 Multi-head self-attention\n")
    A("In **self**-attention queries, keys and values all come from the same sequence $`X`$: every "
      "token asks (query) “which other tokens are relevant to me?”, compares itself against every "
      "token's key, and gathers a weighted mix of their values.\n")
    A(math(r"\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V"))
    A("\n#### A.4.1 Parameters\n")
    doc_weights(md, ENC["attn"], [("Q", "W^Q"), ("K", "W^K"), ("V", "W^V"), ("O", "W^O")])
    doc_attention(md, "A.4", S["enc_attn"], ENC["attn"], X, X, "X", "X", ENC_TOK, ENC_TOK, enc_wr)
    A("Each output row is a new representation of a token that blends in information from the "
      "tokens it attended to — with **two heads** looking at different subspaces, each token can "
      "attend to different partners for different reasons (compare the strongest links of head 1 "
      "and head 2 above).\n")

    A("### A.5 Add & Norm (1)\n")
    doc_add_norm(md, S["enc_ln1"], "X", r"\operatorname{MultiHead}(X)", X, S["enc_attn"]["out"],
                 "X_1", ENC_TOK, enc_wr)

    X1 = S["enc_ln1"]["out"]
    A("### A.6 Position-wise feed-forward network\n")
    doc_ffn(md, S["enc_ffn"], ENC, "X_1", X1, ENC_TOK, enc_wr)

    A("### A.7 Add & Norm (2) → encoder output $`M`$\n")
    doc_add_norm(md, S["enc_ln2"], "X_1", r"\operatorname{FFN}(X_1)", X1, S["enc_ffn"]["out"],
                 "M", ENC_TOK, enc_wr)
    M = S["enc_ln2"]["out"]
    A("$`M`$ is the encoder output, often called the **memory**: one context-aware 4-dimensional "
      "vector per source token. With $`N = 6`$ layers (as in the paper) $`M`$ would be fed into an "
      "identical layer with its own weights, five more times. Every decoder layer reads this same "
      "$`M`$ through cross-attention.\n")

    # ---------------------------------------------------------------- decoder
    A("---\n\n# Part B — Decoder\n")
    A("### B.1 Decoder input (shifted right), embeddings and positions\n")
    A(f"The decoder input is “{' '.join(DEC_TOK)}” (positions 0–3). Embedding and positional "
      "encoding work exactly as in A.1–A.3, with the same shared $`E`$ and the same $`PE`$ formula.\n")
    doc_embedding(md, S["dec_in"], DEC_TOK, 1, "Scaled decoder")
    Y = S["dec_in"]["x"]
    A("Adding $`PE`$ for positions 0–3 (first four rows of the table in A.2):\n")
    A(math(r"Y = \sqrt{d_{\text{model}}}\; E[\text{decoder tokens}] + PE"))
    A("\n**Y** (4×4):\n")
    A(table(Y, DEC_TOK, DIMS))
    A("\n(Because the decoder input here consists of the same words at the same positions as the "
      "first four encoder tokens, $`Y`$ equals the first four rows of $`X`$ — a coincidence of our "
      "toy task, not a property of Transformers.)\n")

    A("### B.2 Masked multi-head self-attention\n")
    A("Decoder self-attention is computed exactly like encoder self-attention, but with **its own "
      "weights** and a **causal mask**: position $`i`$ may only attend to positions $`j \\le i`$. "
      "Otherwise, during training, the token at position 1 could simply look at position 2 "
      "and copy the word it is supposed to predict. The mask is added to the scaled scores "
      "before the softmax:\n")
    mask = np.where(np.triu(np.ones((len(DEC_IDS), len(DEC_IDS)), dtype=bool), 1), -np.inf, 0.0)
    A(table(mask, DEC_TOK, DEC_TOK, d=0))
    A("\nSince $`e^{-\\infty} = 0`$, masked positions receive exactly zero attention.\n")
    A("#### B.2.1 Parameters\n")
    doc_weights(md, DEC["self"], [("Q", r"W^Q_{\text{self}}"), ("K", r"W^K_{\text{self}}"),
                                  ("V", r"W^V_{\text{self}}"), ("O", r"W^O_{\text{self}}")])
    doc_attention(md, "B.2", S["dec_self"], DEC["self"], Y, Y, "Y", "Y", DEC_TOK, DEC_TOK,
                  self_wr, causal=True)
    A(f"Note that the first row (“{DEC_TOK[0]}”) always puts weight 1 on itself — it can see "
      "nothing else.\n")

    A("### B.3 Add & Norm (1)\n")
    doc_add_norm(md, S["dec_ln1"], "Y", r"\operatorname{MaskedMultiHead}(Y)", Y, S["dec_self"]["out"],
                 "Y_1", DEC_TOK, self_wr)
    Y1 = S["dec_ln1"]["out"]

    A("### B.4 Encoder–decoder (cross) attention\n")
    A("This is where the decoder reads the source sentence. The **queries come from the decoder** "
      "($`Y_1`$), the **keys and values come from the encoder output** $`M`$. No mask is needed: "
      "the whole source sentence is known. The score matrix is therefore 4×5 "
      "(4 decoder positions × 5 encoder positions).\n")
    A(math(r"Q = Y_1 W^Q,\qquad K = M\, W^K,\qquad V = M\, W^V"))
    A("\nReminder — encoder output **M** (from A.7):\n")
    A(table(M, ENC_TOK, DIMS))
    A("\n#### B.4.1 Parameters\n")
    doc_weights(md, DEC["cross"], [("Q", r"W^Q_{\text{cross}}"), ("K", r"W^K_{\text{cross}}"),
                                   ("V", r"W^V_{\text{cross}}"), ("O", r"W^O_{\text{cross}}")])
    doc_attention(md, "B.4", S["dec_cross"], DEC["cross"], Y1, M, "Y_1", "M", DEC_TOK, ENC_TOK, cross_wr)
    A("The cross-attention weights are much flatter than in self-attention: the rows of $`M`$ "
      "are similar to each other (an untrained encoder has not yet learned to make the tokens "
      "distinguishable), so the keys are similar too. Training would sharpen these weights so "
      "that, e.g., the position that must output “philosopher” attends to “philosopher” in "
      "the source.\n")
    A("$`K`$ and $`V`$ depend only on the encoder, so at inference time they are computed once per "
      "source sentence and reused for every generated token.\n")

    A("### B.5 Add & Norm (2)\n")
    doc_add_norm(md, S["dec_ln2"], "Y_1", r"\operatorname{CrossAttn}(Y_1, M)", Y1, S["dec_cross"]["out"],
                 "Y_2", DEC_TOK, cross_wr)
    Y2 = S["dec_ln2"]["out"]

    A("### B.6 Position-wise feed-forward network\n")
    A("Same structure as A.6, with the decoder's own weights.\n")
    doc_ffn(md, S["dec_ffn"], DEC, "Y_2", Y2, DEC_TOK, cross_wr)

    A("### B.7 Add & Norm (3) → decoder output\n")
    doc_add_norm(md, S["dec_ln3"], "Y_2", r"\operatorname{FFN}(Y_2)", Y2, S["dec_ffn"]["out"],
                 "Y_3", DEC_TOK, cross_wr)
    Y3 = S["dec_ln3"]["out"]

    A("### B.8 Final linear layer and softmax\n")
    A("The decoder output is mapped to one score (**logit**) per vocabulary word. With weight tying "
      "the projection matrix is $`E^{\\top}`$ (4×5), so the logit of word $`w`$ is simply the dot "
      "product of the decoder vector with the embedding of $`w`$:\n")
    A(math(r"\text{logits} = Y_3\, E^{\top},\qquad P = \operatorname{softmax}(\text{logits})"))
    A(f"\nWorked row **{DEC_TOK[cross_wr]}** (position {cross_wr}; it should predict "
      f"“{VOCAB[TARGET_IDS[cross_wr]]}”):\n")
    lines = [rf"\ell_{{{tok(w)}}} &= {dot(Y3[cross_wr], E[v], db=1)} = {f(S['logits'][cross_wr, v])}"
             for v, w in enumerate(VOCAB)]
    A(aligned(lines))
    A("\n**Logits** (4×5):\n")
    A(table(S["logits"], DEC_TOK, VOCAB))
    ex, sm = S["out_exps"][cross_wr], S["out_sums"][cross_wr]
    A(f"\nSoftmax, worked row **{DEC_TOK[cross_wr]}**:\n")
    A(aligned([
        r"&" + r",\quad ".join(rf"e^{{{f(S['logits'][cross_wr, v])}}} = {f(ex[v])}" for v in range(3)) + ",",
        r"&" + r",\quad ".join(rf"e^{{{f(S['logits'][cross_wr, v])}}} = {f(ex[v])}" for v in range(3, 5)),
        rf"&\textstyle\sum = {f(sm)},\qquad P_{{{tok(DEC_TOK[cross_wr])}}} = {vec(S['probs'][cross_wr])}",
    ]))
    A("\n**Predicted next-token distribution** $`P`$ (row = decoder position, column = candidate word):\n")
    A(table(S["probs"], [f"after “{t}”" for t in DEC_TOK], VOCAB))
    preds = S["probs"].argmax(axis=1)
    A("\n| position | input token | target | argmax prediction | correct? |\n|:-:|:--|:--|:--|:-:|")
    A("\n".join(f"| {i} | {DEC_TOK[i]} | {VOCAB[TARGET_IDS[i]]} | {VOCAB[preds[i]]} | "
                f"{'✓' if preds[i] == TARGET_IDS[i] else '✗'} |" for i in range(len(DEC_IDS))))
    A("\nWith random, untrained weights the predictions are mostly wrong — which is exactly why "
      "we need a loss and training.\n")

    A("### B.9 Training loss (cross-entropy)\n")
    A("For each decoder position the loss is the negative log-probability assigned to the "
      "correct next token; the sentence loss is the average:\n")
    A(math(r"\mathcal L = -\frac{1}{T}\sum_{t=0}^{T-1} \log P_t(y_t), \qquad T = 4"))
    A("")
    A(aligned([rf"-\ln P_{{{t}}}({tok(VOCAB[TARGET_IDS[t]])}) &= -\ln {f(S['probs'][t, TARGET_IDS[t]])} = {f(S['losses'][t])}"
               for t in range(len(DEC_IDS))]
              + [rf"\mathcal L &= \tfrac14({' + '.join(f(x) for x in S['losses'])}) = {f(S['loss'])}"]))
    A(f"\nFor comparison, a model that guesses uniformly has loss $`\\ln 5 = {f(np.log(5))}`$. "
      "Back-propagation computes the gradient of $`\\mathcal L`$ with respect to every weight "
      "matrix above and nudges each weight downhill. The gradient enters the network through a "
      "famously simple expression — for softmax + cross-entropy, "
      "$`\\partial \\mathcal L_t / \\partial\\, \\text{logits}_t = P_t - \\text{onehot}(y_t)`$:\n")
    yhot = np.eye(len(VOCAB))[TARGET_IDS[cross_wr]]
    A(math(rf"\frac{{\partial \mathcal L_{{{cross_wr}}}}}{{\partial\,\text{{logits}}_{{{cross_wr}}}}} = "
           rf"{vec(S['probs'][cross_wr])} - {vec(yhot, 0)} = {vec(S['probs'][cross_wr] - yhot)}"))
    A("\nTraining pushes the logit of the correct word up (negative gradient entry) and all "
      "other logits down.\n")

    A("### B.10 Inference: autoregressive generation\n")
    A("At training time all four decoder positions are processed **in parallel** (teacher forcing). "
      "At inference time there is no target, so the decoder generates one token at a time:\n")
    A("1. Run the encoder **once** → $`M`$ (Part A).\n"
      "2. Start the decoder with the start token (here “Socrates”), compute $`P`$ for the last "
      "position, pick the next token (greedy: argmax; or sample).\n"
      "3. Append that token to the decoder input and repeat until `<eos>` (or a length limit).\n")
    first = S_first_step["probs"][0]
    same = np.array_equal(first, S["probs"][0])
    A("Thanks to the causal mask, position $`i`$ never sees positions $`> i`$, so the row computed "
      "for a prefix is identical whether or not later tokens are present. Running the decoder on "
      "the one-token input “Socrates” alone gives:\n")
    A(table(first, ["decoder input = “Socrates”"], VOCAB))
    A(f"\n— {'exactly' if same else 'NOT'} the first row of $`P`$ in B.8. Greedy decoding would "
      f"output **“{VOCAB[int(np.argmax(first))]}”** as the first generated word (the target is "
      f"“{VOCAB[TARGET_IDS[0]]}”). This equivalence is what lets a Transformer train on all "
      "positions at once while still being a valid left-to-right generator; it also means the "
      "keys and values of earlier positions can be cached (the **KV cache**) instead of being "
      "recomputed for every new token.\n")

    # ---------------------------------------------------------------- summary
    A("---\n\n## Summary of shapes\n")
    A("| Step | Formula | Shape |\n|:--|:--|:-:|\n"
      "| Encoder input | $`X = 2E[\\text{src}] + PE`$ | 5×4 |\n"
      "| Q, K, V (encoder) | $`XW^Q,\\; XW^K,\\; XW^V`$ | 5×4 each |\n"
      "| Per-head scores | $`Q_iK_i^{\\top}/\\sqrt{2}`$ | 5×5 |\n"
      "| Per-head output | $`\\operatorname{softmax}(\\cdot)\\,V_i`$ | 5×2 |\n"
      "| Multi-head output | $`[Z_1\\,Z_2]\\,W^O`$ | 5×4 |\n"
      "| Add & Norm | $`\\operatorname{LN}(X + \\cdot)`$ | 5×4 |\n"
      "| FFN hidden / output | $`\\operatorname{ReLU}(X_1W_1 + b_1)`$ / $`\\cdot\\, W_2 + b_2`$ | 5×8 / 5×4 |\n"
      "| Encoder output $`M`$ | | 5×4 |\n"
      "| Decoder input | $`Y = 2E[\\text{tgt}] + PE`$ | 4×4 |\n"
      "| Masked self-attn scores | $`Q_iK_i^{\\top}/\\sqrt 2 + \\text{Mask}`$ | 4×4 |\n"
      "| Cross-attn scores | $`(Y_1W^Q)_i (MW^K)_i^{\\top}/\\sqrt 2`$ | 4×5 |\n"
      "| Decoder output | $`Y_3`$ | 4×4 |\n"
      "| Logits / probabilities | $`Y_3E^{\\top}`$, softmax | 4×5 |\n")
    n_params = (E.size + sum(w.size for w in ENC["attn"].values())
                + sum(ENC[k].size for k in ("W1", "b1", "W2", "b2"))
                + sum(w.size for p in ("self", "cross") for w in DEC[p].values())
                + sum(DEC[k].size for k in ("W1", "b1", "W2", "b2")))
    A(f"Total trainable parameters in this toy model: **{n_params}** "
      "(the base model in the paper has about 65 million).\n")

    A("## Exercises\n")
    A("1. Verify row “was” of the encoder attention weights $`A_1`$ (A.4.4) by hand.\n"
      "2. Compute $`PE`$ for position 5. Show that $`PE_5`$ can be obtained from $`PE_4`$ by a "
      "rotation in the (dim 0, dim 1) plane by 1 radian.\n"
      "3. Remove the $`1/\\sqrt{d_k}`$ scaling in head 1 of the encoder. Are the attention "
      "weights more or less peaked? Why does the paper scale (think of $`d_k = 64`$)?\n"
      "4. In B.2, what would the attention weights of position 0 be without the mask? Explain "
      "why training without the mask would make the model useless at inference time.\n"
      "5. Permute the encoder input (e.g. swap “most” and “prominent”) but keep $`PE`$ off. "
      "Show that the encoder outputs are permuted the same way (permutation equivariance). "
      "What changes once $`PE`$ is added?\n"
      "6. Count the multiplications in one encoder layer as a function of the sequence length "
      "$`n`$. Which term grows as $`n^2`$?\n"
      "7. Treat the logits of position 3 as free parameters and take one gradient-descent step "
      "with learning rate 1 using the gradient in B.9. Recompute $`P_3`$ and the loss: did "
      "“philosopher” become more likely?\n")
    return "\n".join(md) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare the rounded hand calculation with full precision")
    args = parser.parse_args()

    rnd = make_rounder(full_precision=False)
    S = forward(rnd)
    S_first_step = forward(rnd, dec_ids=DEC_IDS[:1])
    assert np.array_equal(S_first_step["probs"][0], S["probs"][0]), "causal mask is leaking"

    out = Path(__file__).with_name("README.md")
    out.write_text(build_markdown(S, S_first_step), encoding="utf-8")
    print(f"wrote {out}")

    if args.check:
        F = forward(make_rounder(full_precision=True))
        np.set_printoptions(precision=6, suppress=True)
        print("probabilities, 4-decimal hand calculation:\n", S["probs"])
        print("probabilities, full precision:\n", F["probs"])
        print("max |difference| =", np.abs(S["probs"] - F["probs"]).max())
        print(f"loss: hand {S['loss']:.4f}, full precision {F['loss']:.6f}")


if __name__ == "__main__":
    main()
