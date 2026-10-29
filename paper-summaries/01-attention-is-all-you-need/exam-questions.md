# Exam Questions — *Attention Is All You Need*

Questions a demanding professor might ask to check whether you *understand* the Transformer, not just whether you have memorised it. They range from definitions to derivations, calculations, "spot the flaw" statements and design questions.

**How to use this document:** try each question first, then open the answer. Difficulty: ★ basic · ★★ intermediate · ★★★ advanced.

Companion material: [paper summary](README.md) · [pen-and-paper Transformer](../../transformer-pen-and-paper/).

---

## Contents

* [A. Motivation and big picture](#a-motivation-and-big-picture) (Q1–Q6)
* [B. Scaled dot-product attention](#b-scaled-dot-product-attention) (Q7–Q17)
* [C. Multi-head attention](#c-multi-head-attention) (Q18–Q23)
* [D. Architecture: encoder, decoder and their parts](#d-architecture-encoder-decoder-and-their-parts) (Q24–Q34)
* [E. Positional encoding](#e-positional-encoding) (Q35–Q41)
* [F. Complexity and the case for self-attention](#f-complexity-and-the-case-for-self-attention) (Q42–Q46)
* [G. Training and inference](#g-training-and-inference) (Q47–Q55)
* [H. Results, ablations and critical thinking](#h-results-ablations-and-critical-thinking) (Q56–Q63)
* [I. Quick-fire true or false](#i-quick-fire-true-or-false) (Q64)

---

## A. Motivation and big picture

### Q1. ★ What two limitations of recurrent sequence-to-sequence models motivated the Transformer?

<details><summary>Answer</summary>

1. **Sequential computation.** $`h_t = f(h_{t-1}, x_t)`$: step $`t`$ cannot start before step $`t-1`$ finishes, so the positions of one sequence cannot be processed in parallel. This makes training slow, especially for long sequences, and limits batching because of memory.
2. **Long path lengths.** Information from position 1 must pass through $`n-1`$ recurrent updates to reach position $`n`$. Long paths make long-range dependencies hard to learn (signals and gradients fade).

Self-attention fixes both: all positions are computed at once, and any two positions are connected directly in a single layer.

</details>

### Q2. ★★ Convolutional sequence models (ByteNet, ConvS2S) are already parallel. Why did the authors still prefer self-attention?

<details><summary>Answer</summary>

Convolutions are parallel, but a kernel of width $`k`$ only connects nearby positions. To relate two distant positions you need a stack of layers: $`O(n/k)`$ layers for ordinary convolutions or $`O(\log_k n)`$ for dilated ones. The path between distant words therefore still grows with their distance. Self-attention has a **constant** path length $`O(1)`$: every position attends to every other in one layer. (Convolutions are also more expensive per layer by a factor of $`k`$, $`O(k\cdot n\cdot d^2)`$.)

</details>

### Q3. ★★ "The Transformer contains no convolutions." Is this strictly true?

<details><summary>Answer</summary>

Only in spirit. There are no convolutions that slide over *neighbouring positions*, but the paper itself notes that the position-wise feed-forward network can be described as **two convolutions with kernel size 1**: the same weights are applied independently at every position. What is truly absent is any operation that mixes *neighbouring* positions; all mixing across positions happens through attention.

</details>

### Q4. ★★ Is a Transformer fully parallel at inference time?

<details><summary>Answer</summary>

**The encoder is**: all source tokens are processed at once. **The decoder is not**: it generates autoregressively, so token $`t+1`$ can only be computed after token $`t`$ has been chosen; generating $`m`$ tokens takes $`m`$ sequential decoder passes. Parallelism across target positions exists only during **training**, thanks to teacher forcing (the whole correct target is known) together with the causal mask.

</details>

### Q5. ★ Name the task the paper was evaluated on and the metric used.

<details><summary>Answer</summary>

Machine translation on **WMT 2014 English→German** (~4.5 M sentence pairs) and **English→French** (~36 M pairs), measured with **BLEU**, which scores n-gram overlap with reference translations (higher is better). The big model reached 28.4 BLEU (En→De) and 41.8 BLEU (En→Fr). To show generality, the paper also applied it to **English constituency parsing**.

</details>

### Q6. ★★★ The title says attention is "all you need". Name three non-attention components without which the Transformer would not work well, and say what each contributes.

<details><summary>Answer</summary>

* **Positional encodings:** without them self-attention is permutation-equivariant and cannot tell word order (see Q13).
* **Position-wise feed-forward networks:** attention outputs are weighted *averages* of value vectors; the FFN adds per-token non-linear processing and holds most of the parameters.
* **Residual connections + layer normalisation:** let gradients flow through deep stacks and keep activations in a stable range.

Also important: the softmax output layer, label smoothing, dropout and the learning-rate warm-up. The title is a provocation: attention *replaces recurrence and convolution*, not every other component.

</details>

---

## B. Scaled dot-product attention

### Q7. ★ Write the scaled dot-product attention formula and give the shape of every matrix for $`n`$ queries and $`m`$ key–value pairs.

<details><summary>Answer</summary>

```math
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V
```

| Matrix | Shape |
|:--|:-:|
| $`Q`$ | $`n \times d_k`$ |
| $`K`$ | $`m \times d_k`$ |
| $`V`$ | $`m \times d_v`$ |
| $`QK^\top`$, softmax weights | $`n \times m`$ |
| output | $`n \times d_v`$ |

Queries and keys must have the same width $`d_k`$ (for the dot product); values can have a different width $`d_v`$. In self-attention $`n = m`$; in cross-attention $`n`$ = target length and $`m`$ = source length.

</details>

### Q8. ★★ Compute the attention output for $`q = (1, 1)`$, keys $`k_1 = (1, 0)`$, $`k_2 = (0, 1)`$, $`k_3 = (1, 1)`$ and scalar values $`v = (1, 2, 3)`$.

<details><summary>Answer</summary>

$`d_k = 2`$, so divide by $`\sqrt 2 \approx 1.4142`$.

```math
\begin{aligned}
\text{scores} &= (1,\ 1,\ 2), \qquad \text{scaled} = (0.7071,\ 0.7071,\ 1.4142) \\
e^{\text{scaled}} &= (2.0281,\ 2.0281,\ 4.1133), \qquad \textstyle\sum = 8.1695 \\
\text{weights} &= (0.2483,\ 0.2483,\ 0.5035) \\
\text{output} &= 0.2483\cdot 1 + 0.2483\cdot 2 + 0.5035\cdot 3 = 2.2552
\end{aligned}
```

$`k_3`$ is most aligned with $`q`$, so value 3 gets about half of the weight. Note that $`k_1`$ and $`k_2`$ have equal scores, hence equal weights.

</details>

### Q9. ★★★ Derive why the scores are divided by $`\sqrt{d_k}`$.

<details><summary>Answer</summary>

Assume the components $`q_i, k_i`$ are independent random variables with mean 0 and variance 1. Then each product $`q_i k_i`$ has

```math
\mathbb E[q_i k_i] = \mathbb E[q_i]\,\mathbb E[k_i] = 0, \qquad \operatorname{Var}(q_i k_i) = \mathbb E[q_i^2]\,\mathbb E[k_i^2] = 1 .
```

The dot product is a sum of $`d_k`$ independent such terms, so $`\operatorname{Var}(q\cdot k) = d_k`$ and the standard deviation is $`\sqrt{d_k}`$. For $`d_k = 64`$ the scores are typically around $`\pm 8`$, which pushes softmax into saturation (one weight ≈ 1, the rest ≈ 0) where its gradients are tiny. Dividing by $`\sqrt{d_k}`$ gives

```math
\operatorname{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
```

independently of the dimension.

</details>

### Q10. ★★ A student proposes dividing by $`d_k`$ instead of $`\sqrt{d_k}`$ "to be safe". What happens?

<details><summary>Answer</summary>

The variance of the scaled scores becomes $`d_k / d_k^2 = 1/d_k`$; for $`d_k = 64`$ the standard deviation is $`1/8`$. All scores are then close to 0 and the softmax is almost **uniform**: every token attends to every token about equally, and attention loses its ability to focus. The model can partly compensate by learning larger weights in $`W^Q, W^K`$, but this fights the initialisation. $`\sqrt{d_k}`$ is exactly the factor that keeps the variance at 1, neither saturating nor flattening the softmax.

</details>

### Q11. ★★ Compare dot-product (multiplicative) and additive attention. Why did the authors choose dot-product, and when is additive better?

<details><summary>Answer</summary>

* **Additive attention** computes the score with a small feed-forward network, $`\operatorname{score}(q, k) = w^\top \tanh(W_1 q + W_2 k)`$.
* **Dot-product attention** uses $`q\cdot k`$.

Both have similar theoretical complexity, but dot-product attention is **much faster and more memory-efficient in practice**, because all scores come from one highly optimised matrix multiplication $`QK^\top`$. For small $`d_k`$ the two perform similarly; for large $`d_k`$ *unscaled* dot-product attention falls behind additive attention because of the softmax saturation of Q9. The $`1/\sqrt{d_k}`$ scaling removes that weakness, keeping the speed.

</details>

### Q12. ★★ In self-attention, is the attention-weight matrix symmetric? Is the score matrix $`QK^\top`$?

<details><summary>Answer</summary>

**Neither, in general.** $`(QK^\top)_{ij} = x_i W^Q (W^K)^\top x_j^\top`$; this equals $`(QK^\top)_{ji}`$ only if $`W^Q (W^K)^\top`$ is symmetric, which is not enforced (e.g. $`W^Q \ne W^K`$). And even a symmetric score matrix gives non-symmetric weights after the row-wise softmax, because each row is normalised by a different sum. Intuitively, "*it* attends to *animal*" need not imply "*animal* attends to *it*" equally much: attention is a directed relation.

</details>

### Q13. ★★★ Show that self-attention without positional encodings is permutation-equivariant. Why is that a problem?

<details><summary>Answer</summary>

Let $`P`$ be a permutation matrix and permute the input rows, $`X \to PX`$. Then $`Q \to PQ`$, $`K \to PK`$, $`V \to PV`$ and

```math
\operatorname{softmax}\!\left(\frac{PQ(PK)^\top}{\sqrt{d_k}}\right) PV = \operatorname{softmax}\!\left(P\,\frac{QK^\top}{\sqrt{d_k}}\,P^\top\right) PV = P\,\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) P^\top P V = P \cdot \operatorname{Attention}(Q, K, V).
```

(The row-wise softmax commutes with permuting rows and columns, and $`P^\top P = I`$.) The feed-forward layers, residuals and LayerNorm act per position, so they are equivariant too. Hence shuffling the words just shuffles the outputs: *"dog bites man"* and *"man bites dog"* produce the same set of vectors. Word order must be injected explicitly — the job of positional encodings.

</details>

### Q14. ★ What is masking used for in the decoder, and how is it implemented?

<details><summary>Answer</summary>

To preserve the **auto-regressive property**: when predicting token $`i`$, the decoder must not use tokens $`j > i`$ (they are the answers it is learning to predict). Before the softmax, $`-\infty`$ is added to every score with $`j > i`$:

```math
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \qquad M_{ij} = \begin{cases} 0 & j \le i\\ -\infty & j > i\end{cases}
```

$`e^{-\infty} = 0`$, so future positions get exactly zero weight.

</details>

### Q15. ★★ Why add $`-\infty`$ *before* the softmax instead of setting the forbidden weights to zero *after* it?

<details><summary>Answer</summary>

If you zero weights after the softmax, the remaining weights **no longer sum to 1** (some probability mass was given to the future positions and is now lost), so the output is a shrunken, not a weighted, average — and its size would depend on how much mass happened to fall on the future. Adding $`-\infty`$ before the softmax removes the forbidden positions from the normalisation, so the visible weights form a proper distribution. (Renormalising after zeroing would give the same result, but the additive mask is simpler and fuses into one kernel.)

</details>

### Q16. ★★ Besides the causal mask, which other mask is needed when training on batches of sentences?

<details><summary>Answer</summary>

A **padding mask**. Sentences in a batch have different lengths and are padded to the same length with `<pad>` tokens. Keys at padded positions get $`-\infty`$ scores so that no real token attends to padding (in encoder self-attention, decoder self-attention and cross-attention). In the decoder, the padding mask is combined with the causal mask.

</details>

### Q17. ★★ What does attention output if all keys are identical? What if one score is much larger than the rest?

<details><summary>Answer</summary>

* **Identical keys:** all scores in a row are equal, softmax gives uniform weights $`1/m`$, and the output is the **plain average** of the value vectors — the query has no influence.
* **One dominant score:** softmax approaches a one-hot vector and the output approaches **that single value vector** — a "hard" lookup.

Attention interpolates between these two extremes (averaging and selecting), which is why it is called *soft* attention.

</details>

---

## C. Multi-head attention

### Q18. ★ Write the multi-head attention equations and give $`d_k`$ for the base model.

<details><summary>Answer</summary>

```math
\begin{aligned}
\text{head}_i &= \operatorname{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V)\\
\operatorname{MultiHead}(Q, K, V) &= \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O
\end{aligned}
```

with $`W_i^Q, W_i^K \in \mathbb R^{d_{\text{model}}\times d_k}`$, $`W_i^V \in \mathbb R^{d_{\text{model}}\times d_v}`$ and $`W^O \in \mathbb R^{h d_v\times d_{\text{model}}}`$. Base model: $`h = 8`$, $`d_k = d_v = 512/8 = 64`$.

</details>

### Q19. ★★ Why use several heads instead of one big head? Why is the cost about the same?

<details><summary>Answer</summary>

A single head produces **one** weight distribution per token, so different kinds of relations (syntax, coreference, adjacency, …) get averaged together. With several heads, each projects the tokens into its own subspace and can attend according to a different criterion; the paper says multi-head attention lets the model "jointly attend to information from different representation subspaces at different positions".

Cost: each head works in $`d_k = d_{\text{model}}/h`$ dimensions, so computing $`h`$ heads costs about the same as one head of full width $`d_{\text{model}}`$ — the projection matrices have the same total size ($`h`$ blocks of $`d_{\text{model}}\times d_k`$ = one $`d_{\text{model}}\times d_{\text{model}}`$ matrix).

</details>

### Q20. ★★★ Is 8-head attention with $`d_k = 64`$ mathematically equivalent to one head with $`d_k = 512`$, given the same total projection matrices?

<details><summary>Answer</summary>

**No.** The projections can be identical (the 8 heads use the 8 column blocks of one $`512 \times 512`$ matrix), but the **softmax is applied separately per head**. A single head computes one score $`\sum_{i=1}^{512} q_i k_i`$ and one weight distribution; eight heads compute eight partial sums (one per 64-dimensional block) and **eight different weight distributions**. A non-linear softmax of a sum is not the same as eight softmaxes of its parts, so multi-head attention can express patterns (e.g. head 1 looks at the previous word, head 2 at the subject) that one head cannot.

</details>

### Q21. ★★ How many weights does one multi-head attention layer of the base model have?

<details><summary>Answer</summary>

Four $`512\times512`$ matrices ($`W^Q, W^K, W^V`$ as concatenations over heads, and $`W^O`$):

```math
4 \cdot 512 \cdot 512 = 1\,048\,576 \approx 1.05\text{ M}
```

(plus $`4\cdot 512`$ biases if the projections have biases). The count does **not** depend on the number of heads, because $`h \cdot d_k = d_{\text{model}}`$.

</details>

### Q22. ★★ The ablation varied the number of heads at constant compute. What happened with 1 head, and with 32 heads? Explain both.

<details><summary>Answer</summary>

* **1 head:** 0.9 BLEU worse than 8 heads — a single attention distribution cannot capture several relations at once.
* **32 heads** ($`d_k = 16`$): also worse than 8. Each head now has very few dimensions, so its dot products are a crude similarity measure. Too few heads lack diversity; too many heads lack capacity per head. The paper also found that reducing $`d_k`$ alone hurts quality, suggesting that judging compatibility is not trivial.

</details>

### Q23. ★★ What is the role of $`W^O`$?

<details><summary>Answer</summary>

The concatenated head outputs are just the 8 results placed side by side; each lives in its own subspace. $`W^O`$ (512×512) **mixes the heads' findings** into one $`d_{\text{model}}`$-dimensional vector, letting information found by one head interact with that found by another, and maps the result back into the residual stream. Without it, head $`i`$ could only ever write to dimensions $`64(i-1)`$ to $`64i - 1`$.

</details>

---

## D. Architecture: encoder, decoder and their parts

### Q24. ★ Name the three uses of attention in the Transformer and where the queries, keys and values come from in each.

<details><summary>Answer</summary>

| Attention | Queries | Keys & values | Mask |
|:--|:--|:--|:--|
| Encoder self-attention | encoder (previous layer) | encoder (previous layer) | none (padding only) |
| Decoder (masked) self-attention | decoder | decoder | causal |
| Encoder–decoder attention | decoder | **encoder output** | none (padding only) |

The encoder–decoder ("cross") attention mimics the attention of earlier RNN translation models: each target position can look at every source position.

</details>

### Q25. ★★ For a source sentence of 7 tokens and a target prefix of 5 tokens, what are the shapes of the three attention-weight matrices (per head)?

<details><summary>Answer</summary>

* encoder self-attention: $`7 \times 7`$;
* decoder self-attention: $`5 \times 5`$ (lower-triangular after masking);
* cross-attention: $`5 \times 7`$ (rows = target queries, columns = source keys).

Rows always correspond to queries and columns to keys.

</details>

### Q26. ★★ The decoder output at layer $`\ell`$ attends to the encoder output of which layer?

<details><summary>Answer</summary>

To the output of the **final** (6th) encoder layer, in every decoder layer. The encoder is run completely first; its last layer's output (the "memory") provides the keys and values for the cross-attention in all six decoder layers. (Each decoder layer has its own $`W^K, W^V`$, so each reads the memory differently.)

</details>

### Q27. ★ What does the position-wise feed-forward network compute, and why is it called "position-wise"?

<details><summary>Answer</summary>

```math
\operatorname{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2, \qquad 512 \to 2048 \to 512
```

It is applied to **each position separately and identically** (same weights for every token, different weights in every layer). Attention mixes information *across* positions; the FFN transforms each token's vector *individually*, adding non-linearity and capacity.

</details>

### Q28. ★★ How many parameters does one FFN of the base model have? Compare with the attention sub-layer.

<details><summary>Answer</summary>

```math
\underbrace{512\cdot 2048 + 2048}_{W_1,\ b_1} + \underbrace{2048 \cdot 512 + 512}_{W_2,\ b_2} = 2\,099\,712 \approx 2.1\text{ M}
```

That is about **twice** the ≈ 1.05 M of a multi-head attention sub-layer, so roughly two-thirds of each encoder layer's parameters are in the FFN.

</details>

### Q29. ★★★ Estimate the parameter count of the base Transformer (about 65 M) from its components. Use a 37 000-token shared vocabulary.

<details><summary>Answer</summary>

| Component | Count |
|:--|--:|
| Encoder layer: attention (1.05 M) + FFN (2.10 M) + 2 LayerNorms | ≈ 3.15 M |
| × 6 encoder layers | ≈ 18.9 M |
| Decoder layer: 2 attentions (2.10 M) + FFN (2.10 M) + 3 LayerNorms | ≈ 4.20 M |
| × 6 decoder layers | ≈ 25.2 M |
| Shared embedding / output matrix: $`37\,000 \times 512`$ | ≈ 18.9 M |
| **Total** | **≈ 63 M** |

This matches the paper's ≈ 65 M (the exact figure depends on biases and vocabulary size). Without weight tying the embeddings would cost 2–3 times as much — see Q33.

</details>

### Q30. ★★ Why are residual connections used, and what constraint do they place on the architecture?

<details><summary>Answer</summary>

$`x + \operatorname{Sublayer}(x)`$ lets each sub-layer learn a *correction* to its input rather than a full transformation, and gives gradients an identity path through the network, so deep stacks train well (as in ResNets). **Constraint:** $`x`$ and $`\operatorname{Sublayer}(x)`$ must have the same shape, which is why every sub-layer and embedding outputs vectors of width $`d_{\text{model}} = 512`$.

</details>

### Q31. ★★ Why LayerNorm rather than BatchNorm?

<details><summary>Answer</summary>

LayerNorm normalises **each token vector over its own features**; BatchNorm normalises each feature over the **batch**. For sequence models LayerNorm is preferable because:

* sentences have variable lengths and padding, which makes batch statistics messy;
* it behaves identically at training and test time and works with batch size 1 (important for step-by-step decoding);
* it does not couple different examples in a batch.

</details>

### Q32. ★★★ The paper uses *post-norm*. Write post-norm and pre-norm, and explain which is easier to train and why.

<details><summary>Answer</summary>

```math
\text{post-norm: } x_{\ell+1} = \operatorname{LN}\big(x_\ell + F(x_\ell)\big) \qquad\qquad \text{pre-norm: } x_{\ell+1} = x_\ell + F\big(\operatorname{LN}(x_\ell)\big)
```

In pre-norm the residual path from input to output is a pure sum, never passing through a normalisation, so gradients reach early layers undistorted; deep pre-norm models train stably and are less dependent on learning-rate warm-up. In post-norm every residual sum is re-normalised, which can make gradients near the output large early in training — one reason the paper needs warm-up. Many later models (GPT-2 onward, ViT) use pre-norm.

</details>

### Q33. ★★ Explain the weight sharing between embeddings and the output layer. What condition makes three-way sharing possible?

<details><summary>Answer</summary>

The same matrix serves as the encoder input embedding, the decoder input embedding and (transposed) the pre-softmax linear layer, so a word's logit is the dot product between the decoder state and that word's embedding. This saves parameters (one ~19 M matrix instead of three) and ties the input and output representations of each word together. Three-way sharing requires a **shared source–target vocabulary**, e.g. the joint byte-pair encoding used for English–German.

</details>

### Q34. ★★ Why are embeddings multiplied by $`\sqrt{d_{\text{model}}}`$?

<details><summary>Answer</summary>

Embeddings are typically initialised with small values (e.g. variance $`1/d_{\text{model}}`$, which matters especially because the matrix is shared with the output layer), while sinusoidal positional encodings have entries of size up to 1. Multiplying by $`\sqrt{d_{\text{model}}}`$ ($`\approx 22.6`$ for 512) brings the embeddings to a comparable scale, so the token identity is not swamped by the positional signal when the two are added.

</details>

---

## E. Positional encoding

### Q35. ★ Write the sinusoidal positional encoding. What is the range of wavelengths?

<details><summary>Answer</summary>

```math
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
```

Each dimension pair is a sinusoid with wavelength $`2\pi \cdot 10000^{2i/d_{\text{model}}}`$, forming a **geometric progression from $`2\pi`$ to $`10000\cdot 2\pi`$**. Low dimensions change fast with position (fine position), high dimensions slowly (coarse position).

</details>

### Q36. ★★★ Prove that $`PE_{pos+k}`$ is a linear function of $`PE_{pos}`$, with a matrix that does not depend on $`pos`$.

<details><summary>Answer</summary>

For one frequency $`\omega`$, using the angle-addition formulas:

```math
\begin{aligned}
\sin(\omega(pos+k)) &= \sin(\omega\,pos)\cos(\omega k) + \cos(\omega\,pos)\sin(\omega k)\\
\cos(\omega(pos+k)) &= \cos(\omega\,pos)\cos(\omega k) - \sin(\omega\,pos)\sin(\omega k)
\end{aligned}
\;\Longrightarrow\;
\begin{pmatrix}\sin(\omega(pos+k))\\ \cos(\omega(pos+k))\end{pmatrix} = \underbrace{\begin{pmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{pmatrix}}_{R(\omega k)} \begin{pmatrix}\sin(\omega\,pos)\\ \cos(\omega\,pos)\end{pmatrix}
```

Stacking one such rotation block per frequency gives a block-diagonal matrix that depends only on the offset $`k`$. So "$`k`$ positions further" is the same linear map everywhere, which the authors hypothesised would make it easy to attend by **relative** position.

</details>

### Q37. ★★★ Show that the dot product $`PE_p \cdot PE_q`$ depends only on $`p - q`$.

<details><summary>Answer</summary>

For each frequency pair:

```math
\sin(\omega p)\sin(\omega q) + \cos(\omega p)\cos(\omega q) = \cos\big(\omega(p - q)\big)
```

so $`PE_p \cdot PE_q = \sum_i \cos(\omega_i (p - q))`$ — a function of the distance only, largest at distance 0 and generally decreasing (non-monotonically) as $`|p - q|`$ grows. (Once the encodings pass through the learned projections $`W^Q, W^K`$ this exact property is no longer guaranteed, but the representation is naturally distance-aware.)

</details>

### Q38. ★★ Positional encodings are *added* to the embeddings. Why not concatenate them?

<details><summary>Answer</summary>

Adding keeps the width at $`d_{\text{model}}`$, so no extra parameters or computation are needed in any later layer, and residual connections stay simple. In a high-dimensional space the network can learn to separate the two signals (they can occupy largely different directions). Concatenation would also work but would increase the width everywhere or require shrinking the token embedding.

</details>

### Q39. ★★ Learned vs sinusoidal positional encodings: what did the paper find, and why did it keep sinusoids?

<details><summary>Answer</summary>

Learned positional embeddings gave **nearly identical** BLEU (Table 3, row E). Sinusoids were kept because they might let the model **extrapolate to sequences longer than those seen in training**: a learned embedding table has no entry for unseen positions, whereas the sinusoid formula is defined for every position. (In practice extrapolation turned out to be limited, which motivated later relative and rotary schemes; BERT and ViT used learned embeddings.)

</details>

### Q40. ★★ What would go wrong if every dimension used the same frequency, e.g. $`PE_{(pos, j)} = \sin(pos)`$?

<details><summary>Answer</summary>

Positions would no longer be uniquely identifiable: $`\sin`$ has period $`2\pi \approx 6.28`$, so positions $`pos`$ and $`pos + 2\pi m`$ (approximately, e.g. 0 and 44) would look almost the same, and all dimensions would carry the same redundant information. Using many frequencies works like the hands of a clock or the digits of a number: fast components resolve nearby positions, slow ones disambiguate distant positions.

</details>

### Q41. ★ Without positional information, which of these could the model still do: translate "dog bites man" correctly, or detect that the word "dog" is present?

<details><summary>Answer</summary>

It could detect that *dog* is present (that needs only the set of words), but it could **not** reliably translate "dog bites man" versus "man bites dog": both inputs give the same set of encoder outputs (Q13), so who bites whom is lost. The decoder's causal mask provides a little implicit order information on the target side, but nothing on the source side.

</details>

---

## F. Complexity and the case for self-attention

### Q42. ★ Give the per-layer complexity, number of sequential operations and maximum path length for self-attention, recurrent and convolutional layers.

<details><summary>Answer</summary>

| Layer | Complexity per layer | Sequential ops | Max path length |
|:--|:-:|:-:|:-:|
| Self-attention | $`O(n^2 d)`$ | $`O(1)`$ | $`O(1)`$ |
| Recurrent | $`O(n d^2)`$ | $`O(n)`$ | $`O(n)`$ |
| Convolutional | $`O(k n d^2)`$ | $`O(1)`$ | $`O(\log_k n)`$ |
| Restricted self-attention (window $`r`$) | $`O(r n d)`$ | $`O(1)`$ | $`O(n/r)`$ |

</details>

### Q43. ★★ For $`d = 512`$, compare the cost of self-attention and a recurrent layer for $`n = 50`$ and $`n = 4096`$.

<details><summary>Answer</summary>

| $`n`$ | self-attention $`n^2 d`$ | recurrent $`n d^2`$ | cheaper |
|:-:|--:|--:|:--|
| 50 | $`1.28\times10^6`$ | $`1.31\times10^7`$ | self-attention, ~10× |
| 4096 | $`8.59\times10^9`$ | $`1.07\times10^9`$ | recurrent, ~8× |

Self-attention is cheaper exactly when $`n < d`$. Sentences in translation are usually tens of tokens long, far below 512, so self-attention wins — but for long documents, audio or high-resolution images the quadratic term dominates.

</details>

### Q44. ★★ Why does the *maximum path length* matter for learning?

<details><summary>Answer</summary>

Learning a dependency between two positions requires forward signals and backward gradients to travel between them. The more steps the path has, the more the signal is transformed, diluted or vanishes/explodes. The paper argues that the shorter the paths between any combination of input and output positions, the easier it is to learn long-range dependencies. Self-attention connects every pair in one step.

</details>

### Q45. ★★ What is restricted self-attention and what trade-off does it make?

<details><summary>Answer</summary>

Each position attends only to a neighbourhood of size $`r`$ around itself. Cost drops from $`O(n^2 d)`$ to $`O(r n d)`$ (linear in $`n`$), but the maximum path length grows from $`O(1)`$ to $`O(n/r)`$ because distant positions now need several layers to interact. The paper proposes it for very long sequences as future work; it is the seed of later local/sparse attention models.

</details>

### Q46. ★★ The paper mentions interpretability as a side benefit. Give an example and a caveat.

<details><summary>Answer</summary>

**Example:** the appendix visualisations show heads attending from *making* to the distant *more difficult* (completing "making … more difficult"), and heads that seem to perform anaphora resolution (*its* → *Law*). Different heads clearly learn different, often linguistically meaningful, behaviour.

**Caveat:** attention weights show where information is *read from* in one layer of one head; they are not a faithful explanation of the model's decision. The values are mixed again by $`W^O`$, residuals and later layers, and different attention patterns can produce similar outputs.

</details>

---

## G. Training and inference

### Q47. ★ What is "shifted right" in the decoder input, and why is it needed?

<details><summary>Answer</summary>

During training the decoder receives the target sequence with a start token prepended and the last token dropped: input `<bos> y1 y2 … y(m-1)` with labels `y1 y2 … ym`. Position $`t`$ is therefore asked to predict $`y_t`$ having seen only $`y_{<t}`$. Together with the causal mask this lets all positions be trained **in parallel** (teacher forcing) while matching the left-to-right conditions of generation.

</details>

### Q48. ★★★ Compute the learning rate of the base model at steps 1000, 2000, 4000, 16 000 and 64 000. What pattern do you notice?

<details><summary>Answer</summary>

```math
\text{lrate} = 512^{-0.5}\cdot\min(\text{step}^{-0.5},\ \text{step}\cdot 4000^{-1.5})
```

| step | lrate |
|--:|--:|
| 1 000 | $`1.75\times10^{-4}`$ |
| 2 000 | $`3.49\times10^{-4}`$ |
| 4 000 | $`6.99\times10^{-4}`$ (peak) |
| 16 000 | $`3.49\times10^{-4}`$ |
| 64 000 | $`1.75\times10^{-4}`$ |

Pattern: warm-up is linear (half the peak at half the warm-up), decay is $`\propto 1/\sqrt{\text{step}}`$ (half the peak needs 4× the steps). So step 2000 and step 16 000 have the same rate, as do 1000 and 64 000.

</details>

### Q49. ★★ Why is a warm-up phase helpful, particularly with Adam?

<details><summary>Answer</summary>

At the start, weights are random and Adam's running estimates of the gradient mean and variance are based on very few steps, so they are noisy; the variance estimate can be tiny, making effective step sizes huge. Large updates at this point can push the model into a bad region (or make post-norm Transformers diverge). Starting with a small learning rate and ramping it up gives the statistics time to stabilise; afterwards the rate decays to allow fine convergence.

</details>

### Q50. ★★ What is label smoothing? For a vocabulary of 37 000 and $`\epsilon_{ls} = 0.1`$, what target probability does the correct token receive?

<details><summary>Answer</summary>

Instead of a one-hot target, the model is trained towards a mixture of the one-hot vector and the uniform distribution:

```math
q'(k) = (1 - \epsilon_{ls})\,\mathbb 1[k = y] + \frac{\epsilon_{ls}}{K}
```

Correct token: $`0.9 + 0.1/37\,000 \approx 0.900003`$; every other token: $`\approx 2.7\times10^{-6}`$. (Implementations differ slightly, e.g. dividing by $`K-1`$.) This discourages over-confident, very large logits.

</details>

### Q51. ★★★ Label smoothing *hurt* perplexity but *improved* BLEU. Explain why both happen.

<details><summary>Answer</summary>

Perplexity measures how much probability the model assigns to the reference tokens. With label smoothing the model is trained to put only ~90 % on the correct token, so it is deliberately less certain and its perplexity rises. BLEU, however, depends only on which tokens are **ranked highest** during beam search. Smoothing acts as a regulariser: it keeps logits from growing without bound, keeps the alternatives' probabilities reasonable (useful for beam search) and reduces over-fitting, so the top choices are more often correct.

</details>

### Q52. ★ Where is dropout applied in the Transformer?

<details><summary>Answer</summary>

To the **output of every sub-layer** before it is added to the residual and normalised, and to the **sums of embeddings and positional encodings** in both encoder and decoder. Rate 0.1 for the base model (0.3 for the big En→De model, 0.1 for big En→Fr).

</details>

### Q53. ★★ Describe the decoding procedure used at test time.

<details><summary>Answer</summary>

**Beam search** with beam size 4 and length penalty $`\alpha = 0.6`$ (to stop the search from favouring overly short outputs, since every extra token multiplies in another probability below 1). The maximum output length is input length + 50, with early stopping when possible. In addition, the final model is an **average of the last checkpoints** (5 for base, 20 for big), written at 10-minute intervals.

</details>

### Q54. ★★ Why does checkpoint averaging help?

<details><summary>Answer</summary>

Late in training, SGD-style optimisation bounces around a low-loss region. Averaging the weights of several nearby checkpoints lands closer to the centre of that region, which tends to generalise better — a cheap ensemble with no extra cost at inference. It works because the checkpoints come from one training run, so their weights lie in a connected basin and can be averaged meaningfully (averaging independently trained models would not work).

</details>

### Q55. ★★ How long did training take, and why is this a central selling point?

<details><summary>Answer</summary>

On 8 NVIDIA P100 GPUs: base model 100 k steps in **12 hours**; big model 300 k steps in **3.5 days**. Previous state-of-the-art models needed far more compute (the big Transformer used less than a quarter of the training cost of the previous best En→Fr model). Training cost determines how fast researchers can iterate and how large models can grow — the parallelism of self-attention is what later made models with billions of parameters feasible.

</details>

---

## H. Results, ablations and critical thinking

### Q56. ★ State the main translation results.

<details><summary>Answer</summary>

WMT 2014 newstest, BLEU: **28.4** En→De (big), more than 2 BLEU above all previous models *including ensembles*; **41.8** En→Fr (big), a new single-model state of the art. The base model (27.3 En→De) already beat all previously published models and ensembles on En→De, at a fraction of their training cost.

</details>

### Q57. ★★ Why did the authors include English constituency parsing, and what did it show?

<details><summary>Answer</summary>

To test **generalisation** beyond translation, on a task whose output is strongly structured and much longer than the input. A 4-layer Transformer, with little task-specific tuning, reached 91.3 F1 trained only on the WSJ portion of the Penn Treebank (~40 k sentences) and 92.7 F1 semi-supervised, outperforming most earlier models. This suggested the architecture is general-purpose, not a translation-specific trick.

</details>

### Q58. ★★ Summarise three findings from the ablation table (Table 3) and what they teach.

<details><summary>Answer</summary>

* **Heads:** 1 head is 0.9 BLEU worse; too many heads also hurt → diversity of attention patterns matters, but each head needs capacity.
* **$`d_k`$:** shrinking it hurts → computing compatibility is not trivial.
* **Size and regularisation:** bigger models are better, and dropout is very helpful against over-fitting.
* **Positional encoding:** learned vs sinusoidal makes almost no difference.

</details>

### Q59. ★★★ Why can the encoder use full (bidirectional) self-attention while the decoder cannot? What would happen if you removed the decoder mask during training only?

<details><summary>Answer</summary>

The encoder's input (the source sentence) is fully known at test time, so every source token may look at every other. The decoder generates the target left-to-right, so at test time the future target tokens **do not exist yet**. If the mask were removed during training, position $`t`$ could simply copy $`y_t`$ from the input at position $`t+1`$: training loss would drop to near zero, but the model would learn nothing useful, and at test time — where the future tokens are missing — it would fail completely. This is a classic train/test mismatch.

</details>

### Q60. ★★ Why can past keys and values be cached during generation (the "KV cache")?

<details><summary>Answer</summary>

Because of the causal mask, the representation of position $`i`$ in every decoder layer depends only on positions $`\le i`$. Adding a new token at position $`t+1`$ does not change the keys and values already computed for positions $`1..t`$, so they can be stored and reused; each step only computes the new token's query, key and value. Likewise the cross-attention keys and values depend only on the encoder output and are computed once per source sentence.

</details>

### Q61. ★★ Name three limitations of the original Transformer.

<details><summary>Answer</summary>

* **Quadratic time and memory** in sequence length ($`n\times n`$ attention matrix) → limits long documents, high-resolution images, audio.
* **Sequential decoding** at inference despite parallel training.
* **No built-in inductive bias** for order or locality: everything must be learned from data or injected via positional encodings, and the fixed-length context does not naturally extrapolate to longer inputs.

</details>

### Q62. ★★ Encoder-only, decoder-only and encoder–decoder Transformers: which would you pick for (a) sentiment classification, (b) open-ended text generation, (c) translation? Justify.

<details><summary>Answer</summary>

* **(a) Encoder-only** (e.g. BERT): classification needs a rich representation of a fully known input; bidirectional attention uses both sides of every word.
* **(b) Decoder-only** (e.g. GPT): generation is next-token prediction; the causal mask matches how text is produced.
* **(c) Encoder–decoder** (the original Transformer, T5): the input is fully known (encode bidirectionally) while the output is generated (decode causally); cross-attention connects them. Decoder-only models can also translate by treating source + target as one sequence, which is the common approach in modern large language models.

</details>

### Q63. ★★★ A colleague wants to apply the Transformer to documents of 20 000 tokens. Which parts of the architecture become a problem and what would you propose?

<details><summary>Answer</summary>

The attention matrices: $`20\,000^2 = 4\times10^8`$ entries per head per layer, quadratic in memory and time (Q43). The FFN and projections scale linearly and are fine. Options:

* **restricted / local attention** with window $`r`$ (proposed in the paper), possibly plus a few global tokens;
* **sparse or strided attention** patterns, or hierarchical/chunked processing with recurrence over chunks;
* **positional encodings** that generalise to long sequences (sinusoidal, relative, rotary) — learned absolute embeddings stop at the trained length;
* memory-efficient exact attention kernels that avoid storing the full matrix.

A good answer names the bottleneck ($`O(n^2)`$ attention), the trade-off (shorter paths vs cost) and at least one remedy.

</details>

---

## I. Quick-fire true or false

### Q64. ★–★★ True or false? Justify each in one sentence.

<details><summary>Answers</summary>

1. *Self-attention requires $`O(n)`$ sequential operations.* — **False.** $`O(1)`$: all positions are computed in parallel.
2. *The Transformer's decoder attends to all encoder layers.* — **False.** Every decoder layer attends to the output of the last encoder layer.
3. *Multi-head attention has more parameters than single-head attention of the same $`d_{\text{model}}`$.* — **False.** Same projection sizes, since $`h \cdot d_k = d_{\text{model}}`$.
4. *Without positional encodings the Transformer output does not depend on word order.* — **True** for the encoder (permutation-equivariant); the decoder's causal mask leaks some order on the target side.
5. *Scaling by $`\sqrt{d_k}`$ changes which key gets the highest weight.* — **False.** Dividing all scores by a positive constant preserves their order; it only changes how peaked the softmax is.
6. *Label smoothing improves perplexity.* — **False.** It hurts perplexity but improves accuracy and BLEU.
7. *Attention weights in each row sum to 1.* — **True.** Softmax normalises each row.
8. *The FFN mixes information between positions.* — **False.** It is applied independently at each position; only attention mixes positions.
9. *During training, the decoder processes target tokens one at a time.* — **False.** Teacher forcing plus the causal mask lets all target positions be computed in parallel.
10. *Learned positional embeddings performed much worse than sinusoidal ones.* — **False.** Results were nearly identical.
11. *Self-attention is always cheaper than a recurrent layer.* — **False.** Only when $`n < d`$; for long sequences the $`n^2 d`$ term dominates.
12. *Weight tying uses the same matrix for the input embeddings and the pre-softmax projection.* — **True.** All three (encoder input, decoder input, output) share one matrix.

</details>
