# Exam Questions — *BERT: Pre-training of Deep Bidirectional Transformers*

Questions a demanding professor might ask to test whether you understand how BERT works, why it was designed the way it was, and how it is used in practice. They mix definitions, derivations, calculations, design questions and "spot the flaw" statements.

**How to use this document:** try each question first, then open the answer. Difficulty: ★ basic · ★★ intermediate · ★★★ advanced.

Companion material: [paper summary](README.md) · [pen-and-paper MLM](../../bert-pen-and-paper/MLM.md) · [pen-and-paper NSP](../../bert-pen-and-paper/NSP.md).

---

## Contents

* [A. Motivation](#a-motivation) (Q1–Q6)
* [B. Architecture and input representation](#b-architecture-and-input-representation) (Q7–Q15)
* [C. Masked Language Model](#c-masked-language-model) (Q16–Q25)
* [D. Next Sentence Prediction](#d-next-sentence-prediction) (Q26–Q30)
* [E. Pre-training procedure](#e-pre-training-procedure) (Q31–Q35)
* [F. Fine-tuning and use cases](#f-fine-tuning-and-use-cases) (Q36–Q49)
* [G. Results and ablations](#g-results-and-ablations) (Q50–Q55)
* [H. Critical thinking and design](#h-critical-thinking-and-design) (Q56–Q63)
* [I. Quick-fire true or false](#i-quick-fire-true-or-false) (Q64)

---

## A. Motivation

### Q1. ★ What are the two strategies for applying pre-trained language representations to downstream tasks? Give an example of each.

<details><summary>Answer</summary>

* **Feature-based** (e.g. **ELMo**): run the frozen pre-trained model to get contextual word vectors and feed them as additional features into a separate, task-specific architecture.
* **Fine-tuning** (e.g. **OpenAI GPT**, and BERT): add a minimal task-specific output layer and train **all** pre-trained parameters on the downstream task.

BERT is designed mainly for fine-tuning but also works feature-based (Q54).

</details>

### Q2. ★★ What is the main limitation of GPT-style pre-training that BERT addresses? Give a sentence where it matters.

<details><summary>Answer</summary>

GPT is a **left-to-right** language model: its self-attention is masked so each token sees only the tokens before it, so its representations can never use right-hand context. In *"The **bank** of the river was muddy"* the meaning of *bank* is decided by *river*, which comes later. For token-level tasks like question answering, where each passage word must be interpreted using the whole passage and question, this is especially harmful. The paper argues this unidirectionality is the major limitation, particularly for fine-tuning.

</details>

### Q3. ★★ ELMo uses both directions. Why does the paper still call it only "shallowly" bidirectional?

<details><summary>Answer</summary>

ELMo trains a left-to-right LSTM and a right-to-left LSTM **independently** and concatenates their outputs. Inside each network information flows in one direction only; the two directions meet only at the very end. In BERT every layer's self-attention looks at both sides at once, so left and right context **interact in every layer** — "deeply" bidirectional. The paper adds that concatenating two directional models is twice as expensive and strictly less powerful than a deep bidirectional model.

</details>

### Q4. ★★★ Why can't we simply train a deep bidirectional Transformer with the standard language-modelling objective?

<details><summary>Answer</summary>

The standard objective predicts each token from the other tokens. If attention is bidirectional, the token to be predicted is part of the input, and in a multi-layer network it can reach its own prediction **indirectly**: in layer 1 position $`j`$ attends to position $`i`$ and absorbs word $`w_i`$; in layer 2 position $`i`$ attends back to $`j`$. The word effectively "sees itself", the prediction becomes trivial, and the model learns nothing useful. Masked language modelling removes the target word from the input, so this leak cannot happen.

</details>

### Q5. ★ What is the Cloze task and how does it relate to BERT?

<details><summary>Answer</summary>

The Cloze procedure (Taylor, 1953) is a reading-comprehension test in which words are deleted from a passage and the reader fills in the blanks. BERT's masked language model is exactly this: hide some tokens and predict them from the context on both sides. The paper cites it as inspiration.

</details>

### Q6. ★★ What does BERT stand for, and which word reflects its main contribution?

<details><summary>Answer</summary>

**Bidirectional Encoder Representations from Transformers.** "**Bidirectional**" is the key word: deep bidirectional pre-training via MLM. "Encoder" says it uses only the Transformer encoder; "Representations" says its purpose is to produce contextual vectors reused by downstream tasks.

</details>

---

## B. Architecture and input representation

### Q7. ★ Give the configurations of BERT-Base and BERT-Large.

<details><summary>Answer</summary>

| | Layers $`L`$ | Hidden $`H`$ | Heads $`A`$ | Feed-forward | Params |
|:--|:-:|:-:|:-:|:-:|:-:|
| BERT-Base | 12 | 768 | 12 | 3072 | 110 M |
| BERT-Large | 24 | 1024 | 16 | 4096 | 340 M |

Both use $`d_k = H/A = 64`$ per head and a feed-forward size of $`4H`$.

</details>

### Q8. ★★ Why was BERT-Base given exactly the same size as OpenAI GPT?

<details><summary>Answer</summary>

To make the comparison **controlled**: same number of layers, hidden size and heads, so the main difference is the attention direction (GPT: left-only; BERT: bidirectional) and the pre-training objectives. When BERT-Base beats GPT on every GLUE task, the gain can be attributed to bidirectional pre-training rather than to model size. (Data also differed: GPT used BooksCorpus only; BERT used BooksCorpus + Wikipedia — the paper's ablations control for this, see Q50.)

</details>

### Q9. ★★★ Derive the ~110 M parameters of BERT-Base (vocabulary 30 522, max length 512).

<details><summary>Answer</summary>

| Component | Computation | Count |
|:--|:--|--:|
| Token embeddings | $`30\,522 \times 768`$ | 23.44 M |
| Position embeddings | $`512 \times 768`$ | 0.39 M |
| Segment embeddings + embedding LayerNorm | $`2\times768 + 2\times768`$ | ≈ 0.003 M |
| Per layer: attention | $`4\times768^2 + 4\times768`$ | 2.36 M |
| Per layer: FFN | $`2\times768\times3072 + 3072 + 768`$ | 4.72 M |
| Per layer: 2 LayerNorms | $`4\times768`$ | ≈ 0.003 M |
| × 12 layers | $`12 \times 7.09`$ M | 85.05 M |
| Pooler (dense layer on `[CLS]`) | $`768^2 + 768`$ | 0.59 M |
| **Total** | | **≈ 109.5 M** |

About a fifth of the parameters are the token embeddings; most of the rest are in the 12 encoder layers, two-thirds of each in the FFN.

</details>

### Q10. ★ Describe BERT's input representation.

<details><summary>Answer</summary>

* WordPiece tokens; a `[CLS]` token first; `[SEP]` after each segment: `[CLS] A [SEP] B [SEP]` (or `[CLS] A [SEP]` for one text).
* Each position's input vector is the **sum** of three learned embeddings:

```math
\text{input}_i = E_{\text{token}}(w_i) + E_{\text{segment}}(s_i) + E_{\text{position}}(i)
```

* token: which WordPiece; segment: sentence A or B; position: index 0–511 (learned, not sinusoidal).

</details>

### Q11. ★★ What is the purpose of the segment embeddings, given that `[SEP]` already separates the sentences?

<details><summary>Answer</summary>

`[SEP]` marks the boundary at **one** position, but each token would then have to work out through attention on which side of it it lies. Segment embeddings give **every** token a direct, explicit signal of which sentence it belongs to. This matters for sentence-pair tasks (NLI, QA, NSP), where the model must compare the two parts. Note also that the same word in sentences A and B gets different input vectors.

</details>

### Q12. ★★ What is WordPiece tokenisation and why does BERT use it? What complication does it create for masking?

<details><summary>Answer</summary>

WordPiece builds a vocabulary (30 000 entries in the paper) of frequent words and sub-word units; rare words are split, e.g. *playing* → *play* + *##ing*, *flightless* → *flight* + *##less*. Benefits: a small vocabulary, **no out-of-vocabulary tokens**, and shared pieces across related words.

Complication: masking operates on WordPieces, so sometimes only part of a word is masked (*play [MASK]*), which can be easy to guess from the other piece. The original paper masks individual WordPieces; Google later released "whole word masking" models that mask all pieces of a word together.

</details>

### Q13. ★★ Why are the three embeddings summed and not concatenated?

<details><summary>Answer</summary>

Summing keeps the vector width at $`H`$, so the encoder size and parameter count are unchanged, and it is cheap. In a 768-dimensional space the three signals can occupy largely different directions, and the first attention layer can learn to read them apart. Concatenation would require either a wider model or splitting $`H`$ between the three. (An embedding LayerNorm is applied after the sum.)

</details>

### Q14. ★★ What is the maximum input length and what limits it?

<details><summary>Answer</summary>

**512 tokens.** BERT uses a learned position-embedding table with 512 rows, so there is simply no embedding for position 512 or later. Self-attention's quadratic cost also makes longer inputs expensive. Longer documents must be truncated, split into windows, or handled by models designed for long inputs.

</details>

### Q15. ★★ BERT uses GELU instead of ReLU. Define GELU and describe its shape.

<details><summary>Answer</summary>

```math
\operatorname{GELU}(x) = x\,\Phi(x), \qquad \Phi(x) = P(Z \le x),\ Z \sim \mathcal N(0, 1)
```

(often approximated by $`0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])`$). For large positive $`x`$ it is ≈ $`x`$, for large negative $`x`$ ≈ 0, and in between it smoothly weights the input by the probability that it is positive; it dips slightly below zero for small negative inputs (minimum ≈ −0.17 near $`x \approx -0.75`$). Unlike ReLU it is smooth everywhere, with no kink at 0. BERT followed GPT in using it.

</details>

---

## C. Masked Language Model

### Q16. ★ Write the MLM objective.

<details><summary>Answer</summary>

Let $`\mathcal M`$ be the set of chosen positions and $`\tilde x`$ the corrupted input. With $`T_i`$ the final hidden vector at position $`i`$:

```math
\mathcal L_{\text{MLM}} = -\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M} \log P(x_i \mid \tilde x), \qquad P(\cdot\mid\tilde x) = \operatorname{softmax}(T_i W^\top + b)
```

The softmax is over the whole vocabulary (in the released code $`W`$ is the token-embedding matrix, after a small dense + GELU + LayerNorm transform). Only the chosen positions contribute to the loss.

</details>

### Q17. ★ Describe the 80/10/10 masking procedure.

<details><summary>Answer</summary>

15 % of WordPiece positions are chosen at random. Each chosen token is replaced by:

* `[MASK]` with probability 80 % — *my dog is hairy → my dog is [MASK]*;
* a random token with probability 10 % — *my dog is apple*;
* itself (unchanged) with probability 10 % — *my dog is hairy*.

In every case the model must predict the original token.

</details>

### Q18. ★★ For a 512-token sequence, how many tokens are chosen on average, and how many fall in each case? What fraction of *all* tokens is replaced by a random word?

<details><summary>Answer</summary>

* chosen: $`0.15 \times 512 = 76.8`$;
* `[MASK]`: $`0.8 \times 76.8 = 61.44`$;
* random token: $`0.1 \times 76.8 = 7.68`$;
* unchanged: $`7.68`$.

Random replacement affects $`0.15\times0.1 = 1.5\,\%`$ of all tokens — small enough, the paper argues, not to harm the model's language understanding.

</details>

### Q19. ★★★ Explain the reason for each part of the 80/10/10 rule. What would go wrong with 100 % `[MASK]`?

<details><summary>Answer</summary>

* **Mismatch problem:** `[MASK]` occurs in pre-training but never in fine-tuning or real use.
* With **100 % `[MASK]`**, the model would only need good predictions at `[MASK]` positions and could ignore the quality of the representations of ordinary tokens, since they are never tested. Fine-tuning recovers from this, but the feature-based approach suffers (Q55).
* **10 % random:** the model cannot trust any input token blindly, so it must use context to check every token.
* **10 % unchanged:** biases the representation towards the actual observed word, so the representation of a real (unmasked) token is a good guess of that token, as needed at fine-tuning time.

Because the model never knows which tokens will be predicted or which were changed, it must keep a good contextual representation of **every** token.

</details>

### Q20. ★★ How does MLM differ from a denoising auto-encoder?

<details><summary>Answer</summary>

A denoising auto-encoder corrupts the input and reconstructs the **entire** input. BERT only predicts the **chosen** (masked) tokens; there is no reconstruction loss on the others. This focuses learning on inferring missing information from context.

</details>

### Q21. ★★ Why is MLM less sample-efficient than left-to-right language modelling, and did it matter?

<details><summary>Answer</summary>

A left-to-right LM gets a training signal at **every** token; MLM only at the ~15 % chosen tokens, so each batch yields fewer predictions and more steps may be needed. The paper shows MLM does converge slightly more slowly in pre-training, but its downstream accuracy is better than the left-to-right model's almost immediately and the gap keeps growing. (Later work such as ELECTRA targeted this inefficiency by learning from all tokens.)

</details>

### Q22. ★★★ MLM predicts masked tokens independently. Give an example where this is a weakness.

<details><summary>Answer</summary>

If *New York* is masked as *[MASK] [MASK] is a big city*, BERT predicts each position separately given the corrupted context; it does not model their joint distribution. It might give high probability to *New* at position 1 and to *Angeles* at position 2 (from *Los Angeles*), producing the inconsistent *New Angeles*. Autoregressive models predict tokens one after another conditioned on earlier predictions and avoid this. The weakness is mostly irrelevant for representation learning but matters if you try to generate with BERT.

</details>

### Q23. ★★ Can BERT be used to generate text? Why not naturally?

<details><summary>Answer</summary>

Not naturally. BERT was trained to fill in blanks given the **whole** sequence; it has no left-to-right factorisation $`\prod_t P(w_t\mid w_{<t})`$ that defines the probability of a sentence and supports sampling one token at a time. You can iteratively fill masks, but quality and efficiency are poor compared with decoder models (GPT) or encoder–decoders (T5, BART), and masked predictions are independent (Q22).

</details>

### Q24. ★★ Why are the MLM output weights tied to the input token embeddings?

<details><summary>Answer</summary>

The score of word $`w`$ at a masked position is the dot product between the transformed hidden state and $`w`$'s embedding. Tying (1) saves $`30\,522\times768 \approx 23`$ M parameters, and (2) means a word's input and output representations live in the same space, so learning to predict a word also improves its input embedding. The same idea was used in the original Transformer.

</details>

### Q25. ★★ During pre-training, which positions are fed to the MLM softmax, and why does this matter for compute?

<details><summary>Answer</summary>

Only the ~15 % masked positions. The vocabulary softmax is expensive (a $`768\times30\,522`$ matrix multiplication per position), so gathering just the masked positions saves most of that cost. Unmasked positions still matter indirectly: their representations are computed in every layer and provide the context for the predictions.

</details>

---

## D. Next Sentence Prediction

### Q26. ★ How are NSP training examples constructed, and where is the prediction made?

<details><summary>Answer</summary>

For each example, take a sentence A from the corpus. With probability 50 % sentence B is the sentence that **actually follows** A (label **IsNext**); otherwise B is a **random** sentence from the corpus (label **NotNext**). The input is `[CLS] A [SEP] B [SEP]`, and the prediction is a 2-way softmax on $`C`$, the final hidden vector of `[CLS]` (passed through the pooler: dense + tanh).

</details>

### Q27. ★★ Why was NSP introduced? Which tasks was it supposed to help?

<details><summary>Answer</summary>

Language modelling works at the word level and does not directly capture **relationships between sentences**, which are central to question answering (QA) and natural language inference (NLI). NSP is a cheap, automatically labelled task that forces the model to reason about how two text segments relate. The ablation showed removing NSP hurt QNLI, MNLI and SQuAD.

</details>

### Q28. ★★★ The final model achieves 97–98 % NSP accuracy. Why might a professor be sceptical that NSP teaches deep understanding?

<details><summary>Answer</summary>

Because the NotNext examples are **random** sentences, usually from a different document and topic, the task can largely be solved by **topic matching** (do A and B talk about the same thing?) without understanding discourse coherence. High accuracy on an easy task does not imply useful skills. Later work agreed: RoBERTa found that removing NSP (and packing contiguous text) matched or improved results, and ALBERT replaced it with sentence-order prediction (same two sentences, possibly swapped), which cannot be solved by topic alone.

</details>

### Q29. ★★ Is the `[CLS]` vector of a pre-trained BERT a good general-purpose sentence embedding?

<details><summary>Answer</summary>

Not out of the box. It was trained only for the NSP objective, so without fine-tuning it is not optimised to place semantically similar sentences close together; the paper notes $`C`$ is "not a meaningful sentence representation without fine-tuning". It becomes meaningful after fine-tuning on a task. Dedicated methods (e.g. Sentence-BERT, which fine-tunes BERT with siamese networks) produce good sentence embeddings.

</details>

### Q30. ★★ Write the total pre-training loss. Are MLM and NSP trained on separate inputs?

<details><summary>Answer</summary>

```math
\mathcal L = \mathcal L_{\text{MLM}} + \mathcal L_{\text{NSP}}
```

They are trained **jointly on the same inputs**: each sequence pair is masked for MLM *and* carries an NSP label. The paper's examples show both at once: `[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]` → IsNext.

</details>

---

## E. Pre-training procedure

### Q31. ★ What data was BERT pre-trained on, and why must it be document-level?

<details><summary>Answer</summary>

**BooksCorpus** (800 M words) and **English Wikipedia** (2 500 M words, text passages only — lists, tables and headers removed). A document-level corpus is essential to extract **long contiguous sequences** and real consecutive sentence pairs for NSP; a shuffled sentence-level corpus such as the Billion Word Benchmark provides neither.

</details>

### Q32. ★★ Compute the tokens per batch and the approximate number of epochs.

<details><summary>Answer</summary>

Batch: $`256 \times 512 = 131\,072 \approx 128\,000`$ tokens. Over $`10^6`$ steps that is $`\approx 1.3 \times 10^{11}`$ tokens, and $`1.3\times10^{11} / 3.3\times10^{9} \approx 40`$ epochs over the 3.3 B-word corpus — matching the paper's "approximately 40 epochs". (WordPiece token counts are slightly higher than word counts, so this is approximate.)

</details>

### Q33. ★★ Why did the authors pre-train 90 % of the steps with sequences of length 128 and only the last 10 % with length 512?

<details><summary>Answer</summary>

Self-attention costs $`O(n^2)`$: a length-512 sequence costs $`(512/128)^2 = 16`$ times more attention computation than length 128 (the rest of the model scales linearly, so overall a few times more). Short sequences make most of the training much faster. The final 10 % at length 512 is needed to **learn the position embeddings** for positions 128–511, which short sequences never use.

</details>

### Q34. ★ List BERT's main pre-training hyperparameters.

<details><summary>Answer</summary>

Adam with learning rate $`10^{-4}`$, $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, L2 weight decay 0.01, learning-rate warm-up over the first 10 000 steps then linear decay, dropout 0.1 on all layers, GELU activation, batch 256 sequences, 1 M steps. Hardware: 4 Cloud TPUs (16 chips) for Base, 16 Cloud TPUs (64 chips) for Large, 4 days each.

</details>

### Q35. ★★ Does BERT-Base really need 1 million pre-training steps?

<details><summary>Answer</summary>

Yes, according to the paper's ablation: BERT-Base gains almost **1.0 % accuracy on MNLI** when trained for 1 M steps compared with 500 k steps. More pre-training keeps paying off, a finding later pushed much further by RoBERTa.

</details>

---

## F. Fine-tuning and use cases

### Q36. ★ Describe the general fine-tuning procedure and typical hyperparameters.

<details><summary>Answer</summary>

Initialise with the pre-trained weights, format the task's inputs as single text or text pair, add a small output layer, and train **all** parameters end-to-end on labelled data. Typical settings: batch size 16 or 32, learning rate $`5\times10^{-5}`$, $`3\times10^{-5}`$ or $`2\times10^{-5}`$, 2–4 epochs. All results in the paper take at most 1 hour on a Cloud TPU or a few hours on a GPU.

</details>

### Q37. ★★ How many new parameters are added when fine-tuning BERT-Large for 3-class natural language inference?

<details><summary>Answer</summary>

One classification matrix $`W \in \mathbb R^{K\times H}`$ (plus bias): $`3 \times 1024 = 3072`$ weights (+3 biases) — negligible compared with 340 M pre-trained parameters. Prediction: $`P = \operatorname{softmax}(C W^\top)`$.

</details>

### Q38. ★★ How does BERT handle sentence-pair tasks without a special cross-attention module?

<details><summary>Answer</summary>

Both texts are packed into **one** sequence, `[CLS] A [SEP] B [SEP]`, with segment embeddings A and B. Because self-attention connects every token to every other, tokens of A attend to tokens of B (and vice versa) in every layer — encoding the pair with self-attention *is* bidirectional cross-attention. Earlier models typically encoded the two texts separately and then added a cross-attention layer.

</details>

### Q39. ★★★ Explain how BERT is fine-tuned for SQuAD v1.1, including the scoring of a span.

<details><summary>Answer</summary>

Input: `[CLS] question [SEP] passage [SEP]`. Two new vectors $`S, E \in \mathbb R^H`$ are learned. The probability that passage token $`i`$ starts the answer is

```math
P_i^{\text{start}} = \frac{e^{S\cdot T_i}}{\sum_j e^{S\cdot T_j}},
```

and analogously for the end with $`E`$. Training maximises the log-likelihood of the correct start and end positions. At prediction time the span $`(i, j)`$ with $`j \ge i`$ maximising $`S\cdot T_i + E\cdot T_j`$ is selected. Only $`2H`$ new parameters, independent of the passage length — the model "points" at the answer.

</details>

### Q40. ★★★ How does BERT handle unanswerable questions in SQuAD 2.0?

<details><summary>Answer</summary>

An unanswerable question is treated as having an answer span that starts and ends at `[CLS]`. The "no answer" score is

```math
s_{\text{null}} = S\cdot C + E\cdot C,
```

compared with the best non-null span $`\hat s_{i,j} = \max_{j\ge i} S\cdot T_i + E\cdot T_j`$. A span is predicted only if $`\hat s_{i,j} > s_{\text{null}} + \tau`$, where the threshold $`\tau`$ is chosen on the development set to maximise F1.

</details>

### Q41. ★★ How is BERT used for named-entity recognition, and what happens with words split into several WordPieces?

<details><summary>Answer</summary>

Token-level classification: each final vector $`T_i`$ is fed to a softmax over the tag set (e.g. B-PER, I-LOC, O), with no CRF on top. When a word is split into several WordPieces, the paper uses the representation of the **first sub-token** as the input to the classifier for that word; the other pieces receive no label.

</details>

### Q42. ★★ How is BERT fine-tuned for a multiple-choice task such as SWAG?

<details><summary>Answer</summary>

For each of the $`k = 4`$ candidate continuations, build an input `[CLS] context [SEP] continuation_k [SEP]`. A single learned vector $`V`$ scores each pair from its `[CLS]` vector: $`\text{score}_k = V\cdot C_k`$. A softmax over the four scores gives the choice probabilities. BERT-Large reached 86.3 % test accuracy, 8.3 points above GPT.

</details>

### Q43. ★★ For each task, give the input format and which output vector(s) are used: sentiment analysis, paraphrase detection, extractive QA, POS tagging.

<details><summary>Answer</summary>

| Task | Input | Output used |
|:--|:--|:--|
| Sentiment analysis | `[CLS] sentence [SEP]` | $`C`$ → softmax over classes |
| Paraphrase detection | `[CLS] s1 [SEP] s2 [SEP]` | $`C`$ → 2-way softmax |
| Extractive QA | `[CLS] question [SEP] passage [SEP]` | all $`T_i`$ of the passage with $`S`$ and $`E`$ |
| POS tagging | `[CLS] sentence [SEP]` | every $`T_i`$ → softmax over tags |

The same pre-trained network handles all four; only the output layer and the input formatting change.

</details>

### Q44. ★★ The paper mentions that fine-tuning BERT-Large was sometimes unstable on small datasets. What did the authors do?

<details><summary>Answer</summary>

They ran **several random restarts** (same pre-trained checkpoint, different fine-tuning data shuffling and classifier initialisation) and selected the best model on the development set. Large models fine-tuned on a few thousand examples can land in poor solutions by chance.

</details>

### Q45. ★★ When would you use BERT feature-based instead of fine-tuning?

<details><summary>Answer</summary>

* the task needs a **task-specific architecture** that a Transformer encoder cannot easily express;
* you want to **pre-compute** expensive representations once and run many cheap experiments on top;
* you cannot afford to fine-tune and store a full copy of the model per task.

On CoNLL-2003 NER, concatenating the frozen outputs of the top four layers into a BiLSTM reached 96.1 dev F1 vs 96.4 for fine-tuning — nearly as good.

</details>

### Q46. ★★ You need to find, among a million documents, those most relevant to a query. Is "feed `[CLS] query [SEP] document [SEP]` into BERT" a good plan?

<details><summary>Answer</summary>

Not as the only step. This **cross-encoder** is accurate, because query and document attend to each other, but it needs a full BERT pass **per query–document pair** — a million passes per query. Practical systems first retrieve candidates with a cheap method (BM25 or a **bi-encoder** that embeds query and documents separately, so document vectors can be pre-computed), then **re-rank** the top few hundred with the cross-encoder.

</details>

### Q47. ★★ How would you handle a document of 2000 tokens for classification with BERT?

<details><summary>Answer</summary>

BERT accepts at most 512 tokens (Q14). Options:

* **truncate** (keep the beginning, or beginning + end) — simple, often surprisingly good;
* **sliding windows** of ≤ 512 tokens with overlap, classify each and aggregate (average, max or vote);
* **hierarchical:** encode chunks with BERT, combine the chunk vectors with a small model;
* use a model designed for long inputs.

For SQuAD, passages longer than the limit are processed with overlapping windows ("document stride") in the released code.

</details>

### Q48. ★★ You have 1 000 labelled tweets for sentiment and a domain full of slang. How would you use BERT?

<details><summary>Answer</summary>

1. Start from a pre-trained BERT (Base is a sensible start for 1 000 examples).
2. Optionally **continue MLM pre-training on unlabelled tweets** to adapt to the domain's vocabulary and style.
3. **Fine-tune** with a classification head on `[CLS]`, small learning rate ($`2`$–$`5\times10^{-5}`$), 2–4 epochs, early stopping on a validation split.
4. Because the dataset is small, run several random restarts and pick the best on validation (Q44).

</details>

### Q49. ★★ Name tasks for which BERT is a poor choice and explain why.

<details><summary>Answer</summary>

* **Open-ended text generation, summarisation, translation output:** BERT has no autoregressive decoder (Q23); use GPT-style or encoder–decoder models.
* **Very long documents** without chunking: 512-token limit and quadratic cost.
* **Low-latency, low-resource deployment:** 110–340 M parameters may be too heavy; distilled versions exist.
* **Tasks needing up-to-date world knowledge** beyond its 2018 training text.

</details>

---

## G. Results and ablations

### Q50. ★★ Summarise the pre-training ablation and what each row proves.

<details><summary>Answer</summary>

(BERT-Base architecture, dev sets.)

| Model | MNLI | QNLI | MRPC | SST-2 | SQuAD F1 |
|:--|--:|--:|--:|--:|--:|
| MLM + NSP (BERT-Base) | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| MLM only (no NSP) | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| left-to-right, no NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| left-to-right, no NSP, + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

* **No NSP** hurts sentence-pair tasks (QNLI −3.5).
* **Left-to-right** instead of MLM hurts everything, SQuAD by −10.7 F1 → bidirectionality is the big win. This row is trained on the same data as BERT, so it isolates the objective from data differences with GPT.
* **+ BiLSTM** recovers part of the SQuAD loss but hurts GLUE → adding bidirectionality afterwards is no substitute for deep bidirectional pre-training.

</details>

### Q51. ★★★ Why is SQuAD hit far harder (−10.7 F1) than SST-2 (−0.6) by the left-to-right objective?

<details><summary>Answer</summary>

SQuAD is a **token-level** task: the model must decide for every passage token whether it starts or ends the answer, which depends heavily on the words to its **right** (and on the question); a left-to-right model's token representations have none of that right context. SST-2 is a **sentence-level** classification read from one summary vector; in a left-to-right model the final position has already seen the whole sentence, so less information is missing.

</details>

### Q52. ★★ What did the model-size ablation show, and why was it considered surprising?

<details><summary>Answer</summary>

Larger models (more layers, wider, more heads) gave strictly better accuracy on all tested GLUE tasks, **including MRPC with only 3 600 training examples**. It was already known that scale helps tasks with large training sets (e.g. machine translation). The surprise was that, given sufficient pre-training, very large models also help **small** downstream tasks instead of over-fitting them.

</details>

### Q53. ★ Give BERT's headline results.

<details><summary>Answer</summary>

State of the art on 11 tasks, including: GLUE score **80.5** (+7.7 absolute), MultiNLI **86.7 %** (+4.6), SQuAD v1.1 test F1 **93.2** (+1.5), SQuAD v2.0 test F1 **83.1** (+5.1), SWAG **86.3 %** (+8.3 over GPT). BERT-Base beat GPT on every GLUE task at the same size.

</details>

### Q54. ★★ What does the feature-based NER experiment reveal about where information is stored in BERT's layers?

<details><summary>Answer</summary>

Different layer combinations give different results: the embeddings alone are weak (91.0 dev F1), the last hidden layer alone is decent (94.9), and concatenating the **last four layers** is best (96.1), close to fine-tuning (96.4). Useful information is spread over several upper layers rather than being concentrated in the final one, and combining them helps.

</details>

### Q55. ★★★ The masking ablation compared "100 % `[MASK]`", "80/10/10" and "100 % random". What did it find for fine-tuning vs feature-based use, and why?

<details><summary>Answer</summary>

Fine-tuning is fairly **robust** to the masking strategy, but the **feature-based** approach suffers with 100 % `[MASK]`: there the pre-training/test mismatch is never corrected, because the frozen representations of normal (non-`[MASK]`) tokens were never directly trained to be informative. (NER feature-based dev F1: 94.0 with 100 % `[MASK]` vs 94.9 with 80/10/10.) 100 % random replacement is worse too (e.g. lower MNLI accuracy). The 80/10/10 mix is the most robust across both settings.

</details>

---

## H. Critical thinking and design

### Q56. ★★ Compare BERT, GPT and ELMo along architecture, direction and usage.

<details><summary>Answer</summary>

| | ELMo | OpenAI GPT | BERT |
|:--|:--|:--|:--|
| Network | 2-layer bidirectional LSTM (two separate directions) | Transformer decoder (12 layers) | Transformer encoder (12/24 layers) |
| Context | left and right, combined only at the end | left only | left and right in every layer |
| Pre-training objective | two language models | left-to-right LM | MLM + NSP |
| Usage | feature-based | fine-tuning | fine-tuning (or feature-based) |
| Data | 1B Word Benchmark | BooksCorpus | BooksCorpus + Wikipedia |

</details>

### Q57. ★★ Why are encoder models like BERT well suited to classification while decoder models like GPT are suited to generation?

<details><summary>Answer</summary>

Classification and extraction take a **fully known input**, so the best representation of each token uses context on both sides — the bidirectional encoder provides it. Generation must produce text **left-to-right** where the future does not exist yet; a decoder trained with a causal mask matches exactly how it is used, and its probability factorisation supports sampling.

</details>

### Q58. ★★★ BERT introduces a pre-training/fine-tuning mismatch. Name two sources and how BERT (or later work) addresses them.

<details><summary>Answer</summary>

1. **The `[MASK]` token** never appears at fine-tuning time → the 80/10/10 rule (partial fix); ELECTRA later trained a discriminator on *replaced* tokens so that no artificial token is needed.
2. **Sequence length:** if pre-training used only short sequences, position embeddings for long positions would be untrained → the final 10 % of steps at length 512.
3. (Also: NSP pairs vs single-sentence tasks — handled by the flexible input format with segment A only.)

</details>

### Q59. ★★ Why does the size of the pre-training corpus matter more for BERT than for task-specific models?

<details><summary>Answer</summary>

BERT is meant to learn general-purpose knowledge of language (syntax, semantics, facts) once and transfer it to many tasks with little labelled data. Self-supervised objectives can use **unlimited unlabelled text**, and the paper's results (bigger models and longer training keep helping) indicate the model was far from saturated — more data and more steps give better representations for every downstream task.

</details>

### Q60. ★★ What happens to BERT's fine-tuned predictions on text containing many words never seen in pre-training (e.g. new chemical names)?

<details><summary>Answer</summary>

There are no unknown tokens — WordPiece splits new words into known sub-words or characters — so BERT always produces *some* representation. But the pieces may carry little useful meaning, sequences get longer, and performance drops in specialised domains. Remedies: continued pre-training on domain text (as in BioBERT, SciBERT), or training a domain-specific vocabulary.

</details>

### Q61. ★★ How does BERT inherit from the original Transformer, and what does it change?

<details><summary>Answer</summary>

**Inherits:** the encoder layer (multi-head self-attention, position-wise FFN, residual connections, post-LayerNorm), scaled dot-product attention, weight tying between input and output embeddings.

**Changes:** no decoder; **learned** position embeddings instead of sinusoids; **segment** embeddings; GELU instead of ReLU; special `[CLS]`/`[SEP]`/`[MASK]` tokens; self-supervised pre-training (MLM + NSP) followed by fine-tuning, instead of supervised translation training.

</details>

### Q62. ★★★ RoBERTa later improved on BERT without changing the architecture. Based on BERT's own ablations, which changes would you have predicted?

<details><summary>Answer</summary>

* **Train longer and on more data:** BERT's own finding that 1 M steps beat 500 k (Q35) hinted the model was under-trained.
* **Larger batches** and tuned hyperparameters.
* **Reconsider NSP:** its benefit in BERT's ablation was mixed and the task is easy to solve by topic (Q28); RoBERTa dropped it.
* **Dynamic masking** (new masks every time a sequence is seen) instead of masks fixed once during data preparation.

The lesson for the exam: architecture was not the bottleneck; data, training budget and objectives were.

</details>

### Q63. ★★ How did BERT's ideas carry over to computer vision?

<details><summary>Answer</summary>

* The **`[CLS]` token** idea: the Vision Transformer (ViT) prepends a learnable `[class]` token and classifies from its final state.
* The **same encoder architecture**: ViT-Base/Large copy BERT-Base/Large's configurations.
* **Masked prediction** as self-supervision: ViT tried masked patch prediction; later BEiT and MAE made masked image modelling a leading pre-training method.

See the [ViT summary](../03-vit/).

</details>

---

## I. Quick-fire true or false

### Q64. ★–★★ True or false? Justify each in one sentence.

<details><summary>Answers</summary>

1. *BERT uses the Transformer decoder.* — **False.** It uses only the encoder, without a causal mask.
2. *BERT's position embeddings are sinusoidal.* — **False.** They are learned, one per position up to 512.
3. *All masked-out positions are replaced by `[MASK]`.* — **False.** 80 % `[MASK]`, 10 % random, 10 % unchanged.
4. *The MLM loss is computed over all tokens.* — **False.** Only over the ~15 % chosen positions.
5. *NSP negatives are sentences from the same document in the wrong order.* — **False.** They are random sentences from the corpus (sentence-order swaps came later with ALBERT).
6. *Fine-tuning freezes the pre-trained weights.* — **False.** All parameters are updated; freezing is the feature-based approach.
7. *BERT-Base and GPT have the same number of layers and hidden size.* — **True.** 12 layers, 768 hidden, 12 heads.
8. *A single BERT model can be fine-tuned for both NER and QA.* — **True,** separately: the same pre-trained checkpoint, with a different output layer for each task.
9. *Segment embeddings are only needed for sentence pairs.* — **True** in effect: single-sentence inputs use segment A everywhere.
10. *Removing NSP improved every task in BERT's own ablation.* — **False.** In BERT's ablation it hurt QNLI, MNLI and SQuAD; later work (RoBERTa) found otherwise under a different setup.
11. *BERT-Large always beats BERT-Base, even on small datasets.* — **True** on the reported tasks (e.g. RTE, MRPC), although fine-tuning can be unstable and needs restarts.
12. *The pooled `[CLS]` output is a good sentence embedding without fine-tuning.* — **False.** It is only trained for NSP.

</details>
