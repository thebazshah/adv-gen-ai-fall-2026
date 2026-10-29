# Exam Questions — *An Image is Worth 16x16 Words (Vision Transformer)*

Questions a demanding professor might ask to test whether you understand how the Vision Transformer works, why it behaves the way it does, and when to use it. They mix definitions, calculations, comparisons with CNNs and BERT, design questions and "spot the flaw" statements.

**How to use this document:** try each question first, then open the answer. Difficulty: ★ basic · ★★ intermediate · ★★★ advanced.

Companion material: [paper summary](README.md) · [Transformer exam](../01-attention-is-all-you-need/exam-questions.md) · [BERT exam](../02-bert/exam-questions.md).

---

## Contents

* [A. Motivation](#a-motivation) (Q1–Q5)
* [B. Patch embeddings](#b-patch-embeddings) (Q6–Q13)
* [C. Class token and position embeddings](#c-class-token-and-position-embeddings) (Q14–Q20)
* [D. The encoder and the head](#d-the-encoder-and-the-head) (Q21–Q27)
* [E. Inductive bias and hybrids](#e-inductive-bias-and-hybrids) (Q28–Q33)
* [F. Training, data and results](#f-training-data-and-results) (Q34–Q43)
* [G. Looking inside ViT](#g-looking-inside-vit) (Q44–Q48)
* [H. Use cases and critical thinking](#h-use-cases-and-critical-thinking) (Q49–Q57)
* [I. Quick-fire true or false](#i-quick-fire-true-or-false) (Q58)

---

## A. Motivation

### Q1. ★ What is the central claim of the paper?

<details><summary>Answer</summary>

That reliance on convolutions is not necessary for computer vision: a **pure Transformer applied directly to sequences of image patches** can perform very well on image classification. When pre-trained on large datasets and transferred to mid-sized or small benchmarks (ImageNet, CIFAR-100, VTAB), ViT matches or beats state-of-the-art CNNs while requiring substantially fewer computational resources to train.

</details>

### Q2. ★★ Why is applying self-attention to individual pixels impractical? Quantify for a 224×224 image.

<details><summary>Answer</summary>

Self-attention compares every token with every other, so its cost is quadratic in the number of tokens. With one token per pixel:

```math
224\times224 = 50\,176 \text{ tokens} \;\Rightarrow\; 50\,176^2 \approx 2.5\times10^9 \text{ attention scores per head per layer.}
```

That is far too much memory and computation for realistic input sizes.

</details>

### Q3. ★★ How did earlier work bring attention into vision, and what was the drawback the authors wanted to avoid?

<details><summary>Answer</summary>

Earlier approaches either **combined** attention with CNNs (e.g. attention on top of CNN features) or **replaced** convolutions with specialised attention patterns: attention only within local neighbourhoods, sparse attention, or attention along one axis at a time (axial attention). These work, but the special patterns need complex engineering to run efficiently on hardware accelerators. ViT applies the **standard** Transformer with as few modifications as possible, so existing, highly optimised and scalable NLP implementations can be used directly.

</details>

### Q4. ★ Explain the title "An Image is Worth 16x16 Words".

<details><summary>Answer</summary>

A play on "a picture is worth a thousand words". The image is cut into patches of $`16\times16`$ pixels, and each patch is treated like a **word** (token) of a sentence. A $`224\times224`$ image becomes 196 such "words".

</details>

### Q5. ★★ Why might "use a standard Transformer with minimal changes" be a scientific advantage, not just a convenience?

<details><summary>Answer</summary>

* It tests whether the Transformer's success is **general**, rather than the result of vision-specific engineering.
* It inherits the Transformer's excellent **scaling** behaviour and efficient implementations, so much larger models can be trained.
* It allows **unified architectures** across modalities (text and images processed the same way), which later enabled multimodal models.
* With fewer hand-designed assumptions, what the model learns comes from data, which pays off when data is plentiful (Section E).

</details>

---

## B. Patch embeddings

### Q6. ★ Give the formula for the number of patches and the length of a flattened patch. Compute both for a 224×224 RGB image with $`P = 16`$.

<details><summary>Answer</summary>

```math
N = \frac{HW}{P^2}, \qquad \text{patch length} = P^2 \cdot C
```

$`N = 224^2/16^2 = 14\times14 = 196`$; patch length $`= 16\cdot16\cdot3 = 768`$.

</details>

### Q7. ★★ Fill in the table for RGB images.

| Resolution | $`P`$ | $`N`$ | Patch length | Sequence length with `[class]` |
|:-:|:-:|:-:|:-:|:-:|
| 224 | 32 | ? | ? | ? |
| 224 | 14 | ? | ? | ? |
| 384 | 16 | ? | ? | ? |
| 384 | 32 | ? | ? | ? |

<details><summary>Answer</summary>

| Resolution | $`P`$ | $`N`$ | Patch length | Sequence length |
|:-:|:-:|:-:|:-:|:-:|
| 224 | 32 | $`7^2 = 49`$ | 3072 | 50 |
| 224 | 14 | $`16^2 = 256`$ | 588 | 257 |
| 384 | 16 | $`24^2 = 576`$ | 768 | 577 |
| 384 | 32 | $`12^2 = 144`$ | 3072 | 145 |

$`N`$ grows with the square of the resolution and shrinks with the square of the patch size.

</details>

### Q8. ★★ Show that the linear patch projection is equivalent to a convolution. Give its hyperparameters for ViT-B/16.

<details><summary>Answer</summary>

The projection multiplies every flattened $`16\times16\times3`$ patch by the same matrix $`\mathbf E \in \mathbb R^{768\times D}`$. A convolution with $`D`$ filters of size $`16\times16\times3`$ applied with **stride 16** visits exactly the non-overlapping patches and computes, for each filter, a dot product with the patch — i.e. one column of $`\mathbf E`$. So: kernel $`16\times16`$, stride 16, 3 input channels, $`D = 768`$ output channels, no padding. The output is a $`14\times14\times768`$ feature map, flattened into 196 tokens.

</details>

### Q9. ★★ How many parameters does the patch embedding of ViT-B/16 have?

<details><summary>Answer</summary>

$`\mathbf E`$: $`768 \times 768 = 589\,824`$ weights, plus 768 biases = **590 592**. Coincidentally the input and output widths are both 768 for ViT-B/16 ($`16\cdot16\cdot3 = 768 = D`$).

</details>

### Q10. ★★★ ViT-B/16 vs ViT-B/32 at 224×224: how much more does attention cost, and how much do the MLP and projections cost?

<details><summary>Answer</summary>

Sequence lengths (with `[class]`): 197 vs 50. Attention scores scale with $`n^2`$: $`(197/50)^2 \approx 15.5\times`$ more for B/16. The MLPs and the Q/K/V/O projections act per token, so they scale linearly: $`197/50 \approx 3.9\times`$. Halving the patch size thus makes the model roughly 4× more expensive overall, with attention growing fastest — but B/16 sees finer detail and is more accurate.

</details>

### Q11. ★★ Why is the patch projection described as playing the role of the word-embedding table in NLP? What is the key difference?

<details><summary>Answer</summary>

Both map an input unit (word / patch) to a $`D`$-dimensional vector that enters the Transformer. The difference: words come from a **discrete** vocabulary, so the embedding is a lookup table (one row per word); patches are **continuous** vectors of pixel values with no vocabulary, so a learned **linear map** replaces the table. This also means there is no softmax over a vocabulary at the output and no direct analogue of BERT's MLM without extra design (Q48).

</details>

### Q12. ★ What happens if the image size is not divisible by the patch size?

<details><summary>Answer</summary>

The image must be **resized, cropped or padded** so that its height and width are multiples of $`P`$ (e.g. images are resized to 224 or 384, both divisible by 16). ViT itself has no mechanism for partial patches.

</details>

### Q13. ★★ Does the order in which patches are flattened into a sequence matter?

<details><summary>Answer</summary>

Any fixed order works (ViT uses row-major, left-to-right then top-to-bottom), as long as the **same** order is used for every image, because each sequence position has its own learned position embedding. Since self-attention without position embeddings is permutation-equivariant, the order itself carries no meaning; the position embeddings learn what each index means spatially (Q46).

</details>

---

## C. Class token and position embeddings

### Q14. ★ What is the `[class]` token and where does the idea come from?

<details><summary>Answer</summary>

A learnable vector $`x_{\text{class}}`$ prepended to the sequence of patch embeddings, similar to BERT's `[CLS]` token. It corresponds to no image region; through self-attention it gathers information from all patches, and its final-layer state, after LayerNorm, $`y = \operatorname{LN}(z_L^0)`$, is the image representation fed to the classification head.

</details>

### Q15. ★★ What is the alternative to the `[class]` token, and how did it compare?

<details><summary>Answer</summary>

**Global average pooling (GAP)** of all patch outputs, as is usual in CNNs. In the appendix both performed similarly well, but **only with different learning rates** — using GAP with the `[class]` token's learning rate initially looked much worse. The authors kept the `[class]` token to stay close to the original Transformer/BERT design.

</details>

### Q16. ★★ Why does ViT need position embeddings? What happens without them?

<details><summary>Answer</summary>

Self-attention is permutation-equivariant (and per-token MLPs and LayerNorm are too), so without position information ViT treats the image as an **unordered bag of patches**: shuffling the patches would give the same `[class]` output. Spatial layout (which patch is above which) would be lost. The ablation confirms that having **no** position embedding is clearly worse than any of the variants.

</details>

### Q17. ★★ Which kinds of position embeddings were compared, and what was the result?

<details><summary>Answer</summary>

* **1D** learnable embeddings (one per position in raster order) — the default;
* **2D** embeddings (learned row and column embeddings, combined);
* **relative** position embeddings (based on the offset between patches);
* **none**.

All three encoding types performed **similarly**; none was clearly worse than the others. The authors attribute this to the patch level: at $`14\times14`$ resolution, learning spatial relations is easy whichever encoding is used.

</details>

### Q18. ★★★ Why might 1D position embeddings suffice even though images are 2D?

<details><summary>Answer</summary>

Because the embeddings are **learned**: nothing forces them to encode only a 1D order. With training they come to encode the 2D layout — the paper's visualisation shows patches in the same row or column with similar embeddings and nearby patches more similar than distant ones (Q46). With only 196 positions, there is enough signal in the data to learn this structure, so hand-coding it brings no benefit.

</details>

### Q19. ★★★ A ViT-L/16 is pre-trained at 224×224 and fine-tuned at 384×384. What happens to the sequence length and to the position embeddings?

<details><summary>Answer</summary>

The patch size stays 16, so the grid grows from $`14\times14 = 196`$ to $`24\times24 = 576`$ patches (577 tokens with `[class]`). The Transformer accepts any length, but the pre-trained position embeddings exist only for 196 positions and would be meaningless for the new ones. ViT **interpolates them in 2D**: it reshapes the 196 embeddings into a $`14\times14`$ grid of vectors and resizes it (e.g. bicubic interpolation) to $`24\times24`$, according to their location in the original image. The `[class]` position embedding is kept as is.

</details>

### Q20. ★★ Why interpolate the position embeddings in 2D rather than simply stretching the 1D sequence of 196 vectors to 576?

<details><summary>Answer</summary>

Stretching the 1D sequence would mix up spatial meaning: position 14 in the old sequence is the start of row 2, but in a stretched sequence it would land in the middle of row 1 of the new $`24\times24`$ grid. 2D interpolation keeps each embedding at the **same relative image location** (e.g. "top-left corner" stays top-left). This, together with patch extraction, is the only point where knowledge of the 2D structure is manually injected into ViT.

</details>

---

## D. The encoder and the head

### Q21. ★ Write the four equations defining ViT.

<details><summary>Answer</summary>

```math
\begin{aligned}
z_0 &= [x_{\text{class}};\ x_p^1\mathbf E;\ \dots;\ x_p^N\mathbf E] + \mathbf E_{\text{pos}}, && \mathbf E \in \mathbb R^{(P^2C)\times D},\ \mathbf E_{\text{pos}} \in \mathbb R^{(N+1)\times D}\\
z'_\ell &= \operatorname{MSA}(\operatorname{LN}(z_{\ell-1})) + z_{\ell-1}, && \ell = 1..L\\
z_\ell &= \operatorname{MLP}(\operatorname{LN}(z'_\ell)) + z'_\ell, && \ell = 1..L\\
y &= \operatorname{LN}(z_L^0)
\end{aligned}
```

MSA = multi-head self-attention; the MLP has two layers with a GELU non-linearity; $`z_L^0`$ is the `[class]` token after the last layer.

</details>

### Q22. ★★ How does ViT's layer differ from the original Transformer/BERT layer? Why does it matter?

<details><summary>Answer</summary>

ViT uses **pre-norm**: LayerNorm is applied *before* each sub-layer, $`x + F(\operatorname{LN}(x))`$, whereas the original Transformer and BERT use **post-norm**, $`\operatorname{LN}(x + F(x))`$. In pre-norm the residual stream is a clean sum from input to output, which keeps gradients well-behaved in deep stacks (ViT-Huge has 32 layers) and makes optimisation more stable. Consequently ViT also needs a final LayerNorm on the output, which appears as $`y = \operatorname{LN}(z_L^0)`$.

</details>

### Q23. ★ Give the configurations of ViT-Base, -Large and -Huge, and decode the name "ViT-L/16".

<details><summary>Answer</summary>

| Model | Layers | Hidden $`D`$ | MLP size | Heads | Params |
|:--|:-:|:-:|:-:|:-:|:-:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86 M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307 M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632 M |

Base and Large are taken directly from BERT. **ViT-L/16** = the Large model with $`16\times16`$ input patches.

</details>

### Q24. ★★★ Derive the ~86 M parameters of ViT-B/16.

<details><summary>Answer</summary>

| Component | Count |
|:--|--:|
| Patch projection $`768\times768 + 768`$ | 0.59 M |
| `[class]` token + position embeddings $`(1 + 197)\times768`$ | 0.15 M |
| Per layer: attention $`4\times768^2 + 4\times768`$ | 2.36 M |
| Per layer: MLP $`2\times768\times3072 + 3072 + 768`$ | 4.72 M |
| Per layer: 2 LayerNorms | ≈ 0.003 M |
| × 12 layers | 85.05 M |
| Final LayerNorm (+ head, depends on classes) | small |
| **Total** | **≈ 85.8 M ≈ 86 M** |

Unlike BERT-Base (110 M), there is no 23 M-parameter vocabulary embedding — the patch projection is tiny, which is why ViT-Base is smaller than BERT-Base despite an identical encoder.

</details>

### Q25. ★★ Describe the classification head during pre-training and during fine-tuning. Why is the fine-tuning head zero-initialised?

<details><summary>Answer</summary>

* **Pre-training:** an MLP with one hidden layer on $`y`$.
* **Fine-tuning:** the pre-trained head is removed and replaced by a single **zero-initialised** linear layer $`D \times K`$ for the $`K`$ downstream classes.

Zero initialisation makes the initial output logits all zero, so the model starts with uniform predictions and the first gradient steps do not send large random signals back into the pre-trained network. Only the new layer has to be learned from scratch.

</details>

### Q26. ★★ Does ViT use a causal mask? Why or why not?

<details><summary>Answer</summary>

No. Classification needs every patch to be able to use the whole image, and nothing is generated sequentially, so every patch attends to every other patch (as in BERT). A causal mask would arbitrarily prevent the top of the image from seeing the bottom.

</details>

### Q27. ★★ What non-linearity is used in the MLP, and what is the MLP's role in ViT?

<details><summary>Answer</summary>

**GELU**, $`x\Phi(x)`$, as in BERT. Attention *mixes* information across patches (global interactions); the MLP *processes* each token independently (local, per-patch feature transformation). In fact, the paper notes the MLP layers are among the few parts of ViT that are local and translation-equivariant (Q29).

</details>

---

## E. Inductive bias and hybrids

### Q28. ★ What is an inductive bias? Name the three inductive biases of CNNs discussed in the paper.

<details><summary>Answer</summary>

An **inductive bias** is an assumption built into a model's architecture that guides learning before seeing any data. CNNs build in:

1. **Locality:** each output depends on a small neighbourhood of the input;
2. **Two-dimensional neighbourhood structure:** neighbourhoods are 2D windows;
3. **Translation equivariance:** shifting the input shifts the feature map by the same amount (the same filter is applied everywhere).

These are baked into every layer of a CNN.

</details>

### Q29. ★★ Which of these biases does ViT have, and where?

<details><summary>Answer</summary>

Much less. In ViT:

* only the **MLP layers** (and the patch projection) are local and translation-equivariant, because they act on each patch separately with shared weights;
* **self-attention is global** from the first layer;
* the 2D neighbourhood structure is used **only** when cutting the image into patches and when interpolating position embeddings for a new resolution;
* position embeddings start random and contain **no** information about 2D positions; all spatial relations must be learned.

</details>

### Q30. ★★★ Explain why ViT underperforms ResNets on ImageNet-1k but outperforms them on JFT-300M, using the bias–variance idea.

<details><summary>Answer</summary>

Strong inductive biases restrict the space of functions a model can easily learn. With **little data**, this is helpful: a CNN does not need to discover locality and translation equivariance from examples, so it generalises better (lower variance). ViT, with fewer constraints, can fit many functions that do not generalise and over-fits a 1.3 M-image dataset. With **lots of data**, the data alone are enough to learn the relevant patterns, and the CNN's built-in assumptions become a ceiling (a source of bias) — ViT's flexibility then wins. The paper summarises: *large-scale training trumps inductive bias*.

</details>

### Q31. ★★ Is ViT translation-equivariant? Consider shifting an image by 16 pixels and by 5 pixels.

<details><summary>Answer</summary>

**Not in general.** A shift by exactly one patch (16 px) moves each patch to a neighbouring grid cell, but each patch now receives a **different position embedding**, so the representations change unless the model has learned to be invariant. A shift by 5 px changes the *content* of every patch (pixels move across patch boundaries), so even the patch embeddings change. Any translation robustness ViT shows is **learned** from data (and augmentation), not built in like in a CNN.

</details>

### Q32. ★★ What is the hybrid architecture and when does it help?

<details><summary>Answer</summary>

Instead of raw image patches, the input sequence is formed from the **feature map of a CNN** (a ResNet): each spatial position of the feature map (patch size $`1\times1`$) is projected to $`D`$ and fed to the Transformer. The CNN supplies local features and some inductive bias. Hybrids slightly **outperform** pure ViT at **small computational budgets**, but the difference **vanishes** for larger models — surprising, since one might expect convolutional local features to help at every size.

</details>

### Q33. ★★ In the hybrid, the paper notes that localised attention heads are less pronounced. What does that suggest?

<details><summary>Answer</summary>

In pure ViT, some heads in the lowest layers attend only locally — they act like the early layers of a CNN. In the hybrid, a ResNet already computes local features before the Transformer, so those local heads are less needed and appear less. This suggests the local heads in pure ViT may serve a similar function to early convolutional layers: ViT *learns* a form of locality when it is not given one.

</details>

---

## F. Training, data and results

### Q34. ★ Name the three pre-training datasets and their sizes.

<details><summary>Answer</summary>

| Dataset | Images | Classes |
|:--|--:|--:|
| ImageNet-1k (ILSVRC-2012) | 1.3 M | 1 000 |
| ImageNet-21k | 14 M | 21 000 |
| JFT-300M (Google internal) | 303 M | 18 000 |

Pre-training datasets were **de-duplicated** with respect to the test sets of the downstream tasks, to avoid test leakage.

</details>

### Q35. ★★ Describe the optimisation setup for pre-training and fine-tuning.

<details><summary>Answer</summary>

* **Pre-training:** Adam ($`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$), batch size 4096, high weight decay 0.1, linear learning-rate warm-up and decay, resolution 224. Adam worked slightly better than SGD here, even for the ResNet baselines.
* **Fine-tuning:** SGD with momentum 0.9, batch size 512, usually at a higher resolution than pre-training.

The strong weight decay and other regularisation matter, particularly for the smaller pre-training datasets.

</details>

### Q36. ★ State ViT's headline results.

<details><summary>Answer</summary>

ViT-H/14 pre-trained on JFT-300M: **88.55 %** ImageNet top-1, **90.72 %** ImageNet-ReaL, **94.55 %** CIFAR-100, **77.63 %** on VTAB (19 tasks) — matching or beating BiT-L (87.54 % on ImageNet) and Noisy Student (88.4/88.5 %), with **2.5 k** TPUv3-core-days of pre-training compute versus 9.9 k for BiT-L and 12.3 k for Noisy Student.

</details>

### Q37. ★★ By what factor is ViT-H/14 cheaper to pre-train than BiT-L, and ViT-L/16 (JFT) cheaper than BiT-L? Is the comparison entirely fair?

<details><summary>Answer</summary>

$`9.9\text{k}/2.5\text{k} \approx 4\times`$ and $`9.9\text{k}/0.68\text{k} \approx 15\times`$. Caveat: compute is measured in **TPUv3-core-days**, which reflects hardware efficiency as well as FLOPs; Transformers map particularly well onto TPUs (large dense matrix multiplications). Training budgets and recipes also differ between methods. The paper's controlled scaling study (Q41) gives a fairer comparison: ~2–4× less compute for the same accuracy.

</details>

### Q38. ★★ What is ViT's result when pre-trained on the public ImageNet-21k, and why is it significant?

<details><summary>Answer</summary>

ViT-L/16 pre-trained on ImageNet-21k reaches **85.30 %** on ImageNet (88.62 % ReaL, 93.25 % CIFAR-100), trainable on a standard cloud TPUv3 with 8 cores in about 30 days. It shows strong results are possible **without** private data like JFT-300M, making the approach accessible to the wider research community.

</details>

### Q39. ★★★ When pre-trained on ImageNet-1k only, ViT-Large is *worse* than ViT-Base. Explain.

<details><summary>Answer</summary>

With only 1.3 M images, the larger model's extra capacity mostly **over-fits**, despite moderate regularisation (weight decay, dropout, label smoothing). ViT-Base, with fewer parameters, is constrained more and generalises better. With ImageNet-21k the two are similar, and only with JFT-300M does the Large model show its full benefit. Model size must be matched to data size — especially for a model with weak inductive biases.

</details>

### Q40. ★★ Describe the JFT-subset experiment and its conclusion.

<details><summary>Answer</summary>

Models were pre-trained on random subsets of JFT of 9 M, 30 M, 90 M and the full 300 M images, **without extra regularisation** (to isolate the effect of data size, using the same hyperparameters for all subsets and early stopping), and evaluated by **few-shot linear accuracy**. ViT over-fits more than ResNets of comparable compute on small subsets: e.g. ViT-B/32 is much worse than ResNet50 on 9 M but better from 90 M on. Conclusion: the convolutional inductive bias is useful for smaller datasets, but for larger ones learning the relevant patterns directly from data is sufficient, even beneficial.

</details>

### Q41. ★★ Summarise the three findings of the scaling (performance vs compute) study.

<details><summary>Answer</summary>

1. **ViT dominates ResNets on the performance/compute trade-off**, using roughly **2–4× less compute** for the same transfer accuracy (averaged over five datasets).
2. **Hybrids** slightly beat pure ViT at small compute budgets, but the difference vanishes for larger models.
3. **ViT does not appear to saturate** within the range tried, motivating further scaling.

</details>

### Q42. ★★ What is VTAB and why is it a good test for representation quality?

<details><summary>Answer</summary>

The **Visual Task Adaptation Benchmark**: 19 diverse tasks with only **1 000 training examples each**, grouped into *Natural* (ordinary photos, e.g. CIFAR, Pets), *Specialized* (medical and satellite imagery) and *Structured* (geometric understanding such as counting or localisation). Because training data are scarce and tasks differ from the pre-training data, VTAB measures how well a representation **transfers** in a low-data regime. ViT-H/14 beat the previous best on Natural and Structured tasks and was on par on Specialized.

</details>

### Q43. ★★ How many few-shot examples per class is "linear few-shot evaluation", and why was it used?

<details><summary>Answer</summary>

In few-shot linear evaluation the pre-trained model is **frozen** and a linear classifier (solved in closed form as regularised least-squares regression) is fitted on a small subset of the training images, typically **5 or 10 examples per class** ("5-shot", "10-shot"). It is used when full fine-tuning would be too costly to run for every model and dataset in a large study, e.g. the JFT-subset experiment. It measures how linearly separable the frozen representations are.

</details>

---

## G. Looking inside ViT

### Q44. ★★ What do the learned patch-embedding filters look like?

<details><summary>Answer</summary>

The top principal components of the projection $`\mathbf E`$ resemble **plausible basis functions** for describing the fine structure within each patch — edges, colour blobs and smooth gradients, similar to Gabor-like first-layer filters of CNNs. So the first layer learns something CNN-like even though it is a single linear map.

</details>

### Q45. ★★★ Define "mean attention distance". What did it reveal?

<details><summary>Answer</summary>

For each head and query patch, take the **pixel distance** between the query patch and every other patch, weighted by the attention weight, and average over many images: the typical distance across which a head integrates information — the analogue of a CNN's **receptive field size**.

Findings:

* Some heads attend to **most of the image already in the lowest layers**: ViT integrates global information from the start, which a CNN cannot do (its receptive field grows slowly with depth).
* Other low-layer heads have small distances (**local** heads, CNN-like).
* Attention distance **increases with depth**; in the deepest layers almost all heads are global.

</details>

### Q46. ★★ What do the learned position embeddings reveal when compared by cosine similarity?

<details><summary>Answer</summary>

Each position embedding is most similar to itself and to **nearby** positions (distance is encoded), and there is a clear **row–column structure**: patches in the same row or column have similar embeddings. For larger grids a **sinusoidal** structure sometimes appears. ViT learns the 2D image topology from data, explaining why hand-crafted 2D embeddings gave no improvement.

</details>

### Q47. ★★ What does attention rollout show?

<details><summary>Answer</summary>

Attention rollout combines (multiplies) the attention matrices of all layers, accounting for residual connections, to estimate how much each input patch contributes to the `[class]` output. Visualised as a map over the image, it shows that ViT attends to regions that are **semantically relevant** for classification (e.g. the dog in a dog photo). A useful qualitative sanity check — but, as with all attention visualisations, not a rigorous explanation.

</details>

### Q48. ★★★ Describe the masked patch prediction experiment. How is it related to BERT, and what was the result?

<details><summary>Answer</summary>

Inspired by BERT's MLM: **50 %** of the patch embeddings are corrupted (replaced by a learnable `[mask]` embedding 80 % of the time, by a random other patch embedding 10 %, kept 10 %), and the model predicts, for each corrupted patch, a coarse description of its content — the **3-bit mean colour** (512 classes) — from its final representation. Pixels are continuous, so a discrete target had to be designed. ViT-B/16 pre-trained this way reached **79.9 %** on ImageNet: about **2 %** better than training from scratch but still about **4 %** behind supervised pre-training. The authors left contrastive pre-training to future work; later methods (BEiT, MAE) made masked image modelling highly effective.

</details>

---

## H. Use cases and critical thinking

### Q49. ★★ You have 5 000 labelled medical images. Should you train ViT from scratch? What would you do instead?

<details><summary>Answer</summary>

**No** — ViT trained from scratch on so little data will over-fit and lose to a CNN (Q30, Q39). Better options:

* start from a ViT **pre-trained on a large dataset** (ImageNet-21k or larger) and **fine-tune**, replacing the head with a zero-initialised linear layer, possibly at higher resolution;
* use strong data augmentation and regularisation;
* consider a **CNN** or **hybrid** model, whose inductive biases help in the small-data regime;
* if unlabelled domain images are available, consider **self-supervised pre-training** on them.

</details>

### Q50. ★★ Why was the original ViT not directly applied to object detection or segmentation? What would be needed?

<details><summary>Answer</summary>

Dense prediction needs **high-resolution** inputs and **fine spatial detail**. ViT produces a single-scale feature map at $`1/16`$ resolution, and attention cost grows quadratically with the number of patches (Q51). Needed: multi-scale (hierarchical) features, efficient attention for many tokens (e.g. windowed attention as in the later Swin Transformer), and a detection/segmentation head. The paper names applying ViT to detection and segmentation as a challenge for future work.

</details>

### Q51. ★★ A 1024×1024 image is processed with $`P = 16`$. How many tokens and attention scores per head per layer?

<details><summary>Answer</summary>

$`N = (1024/16)^2 = 64^2 = 4096`$ tokens (4097 with `[class]`). Attention scores $`\approx 4096^2 \approx 1.7\times10^7`$ per head per layer — about 450× more than at 224×224 (38 416). This quadratic growth is why high-resolution vision needs efficient attention variants.

</details>

### Q52. ★★★ Compare ViT and BERT: list three similarities and three differences.

<details><summary>Answer</summary>

**Similarities:** Transformer encoder with the same Base/Large configurations; a special classification token (`[CLS]` / `[class]`) whose output represents the input; learned 1D position embeddings; GELU; pre-train on a large dataset then fine-tune.

**Differences:**

* **Input:** discrete WordPiece tokens with an embedding table vs continuous pixel patches with a linear projection.
* **Pre-training:** self-supervised MLM + NSP vs **supervised** classification on labelled images (masked patch prediction only explored briefly).
* **Normalisation:** post-norm (BERT) vs pre-norm (ViT).
* **Segment embeddings:** BERT has them; ViT has no notion of segments.

</details>

### Q53. ★★ Why might ViT's ability to attend globally in early layers be an advantage over CNNs? Give an example.

<details><summary>Answer</summary>

A CNN needs many layers before distant regions of the image can interact (its receptive field grows gradually). ViT can relate any two patches in the **first** layer. Example: recognising that two distant parts belong to the same object (the head and tail of a long animal), or using global context (sky at the top suggests an outdoor scene) to interpret a small, ambiguous region early in processing.

</details>

### Q54. ★★ Many modern vision-language models (e.g. CLIP) use a ViT as the image encoder. Why is ViT a convenient choice?

<details><summary>Answer</summary>

* ViT turns an image into a **sequence of tokens**, the same interface as text Transformers, which simplifies combining modalities (e.g. cross-attention between image tokens and text tokens).
* It **scales** well with data and compute, which matters for web-scale image–text datasets.
* The `[class]` output gives a natural single image embedding for contrastive matching with text embeddings.
* The same efficient Transformer software and hardware optimisations serve both modalities.

</details>

### Q55. ★★★ The paper argues for "fewer image-specific inductive biases". Critique this position.

<details><summary>Answer</summary>

**For:** with enough data, learned structure beats hand-designed structure; ViT matches CNNs at lower compute and keeps improving with scale; a uniform architecture simplifies multimodal work.

**Against / caveats:**

* the conclusion depends on **huge datasets** (JFT-300M is private); in the common small-data regime biases still help;
* later work showed good **training recipes** and distillation (DeiT) let ViT compete on ImageNet-1k alone, so part of the "data hunger" was about optimisation, not just bias;
* successful later models (e.g. Swin, hybrid convolution–attention designs) **re-introduced locality and hierarchy** for efficiency and dense tasks;
* compute comparisons in TPU-days favour Transformer-friendly hardware.

A balanced answer: inductive biases are a trade-off between sample efficiency and flexibility, and the right amount depends on the data and compute available.

</details>

### Q56. ★★ How would you adapt ViT to classify video clips?

<details><summary>Answer</summary>

Treat space–time as a sequence: cut each frame (or short clip) into patches, or use 3D "tubelets" (e.g. $`2\times16\times16`$ pixels across 2 frames), project them linearly, add **spatio-temporal** position embeddings, and run the Transformer encoder with a `[class]` token. The challenge is the number of tokens (frames × patches) under quadratic attention, which can be handled by factorising attention into spatial and temporal parts or by using larger tubelets. (This is the approach taken by later video Transformers.)

</details>

### Q57. ★★ What would you expect to happen if you fine-tuned at a *lower* resolution than pre-training, e.g. 160×160 with $`P = 16`$?

<details><summary>Answer</summary>

The grid shrinks to $`10\times10 = 100`$ patches. The position embeddings must again be 2D-interpolated (down-sampled from $`14\times14`$ to $`10\times10`$). Computation drops (fewer tokens, quadratic attention saving) but each patch covers a larger fraction of the object, so fine detail is lost and accuracy typically drops. The paper's practice is the opposite — fine-tuning at a **higher** resolution than pre-training, which usually improves accuracy at higher cost.

</details>

---

## I. Quick-fire true or false

### Q58. ★–★★ True or false? Justify each in one sentence.

<details><summary>Answers</summary>

1. *ViT contains no convolutional operation at all.* — **False** in effect: the patch projection is equivalent to a convolution with kernel = stride = patch size (and the hybrid uses a full ResNet).
2. *ViT uses a causal attention mask.* — **False.** Every patch attends to every patch.
3. *ViT beats ResNets when both are trained only on ImageNet-1k.* — **False.** ViT is slightly worse in that regime.
4. *Smaller patches mean shorter sequences.* — **False.** $`N = HW/P^2`$: halving $`P`$ quadruples the sequence length.
5. *2D position embeddings clearly outperform 1D ones.* — **False.** The differences were small; only no position embedding was clearly worse.
6. *ViT applies LayerNorm before each sub-layer.* — **True.** It uses pre-norm, unlike the original Transformer and BERT.
7. *The `[class]` token is required; average pooling does not work.* — **False.** GAP works about as well with a suitably tuned learning rate.
8. *ViT is pre-trained with self-supervised masked patch prediction.* — **False.** Main results use supervised pre-training; masked patch prediction was a preliminary experiment.
9. *Changing the resolution at fine-tuning requires retraining the patch projection from scratch.* — **False.** The patch size stays the same; only the position embeddings are interpolated.
10. *Hybrid models are better than pure ViT at all scales.* — **False.** Their advantage at small compute disappears for larger models.
11. *Some attention heads in ViT's first layer attend across most of the image.* — **True.** Shown by the mean attention distance analysis.
12. *ViT-Base has the same encoder configuration as BERT-Base.* — **True.** 12 layers, hidden size 768, 12 heads, MLP size 3072.

</details>
