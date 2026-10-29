# adv-gen-ai-fall-2026

> [!IMPORTANT]
> **Disclaimer**
>
> This repository was created in good faith to provide preparation material for the course. Despite care in preparing it, the worked examples, calculations, summaries and answers may contain mistakes or simplifications. **Please verify the calculations yourself** rather than treating them as authoritative.
>
> The summaries and question-and-answer documents are study aids, not substitutes for the original research papers. They cannot provide the depth of understanding that comes from reading the papers themselves, so please read the originals:
>
> - Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
> - Devlin et al., [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805) (2018)
> - Dosovitskiy et al., [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929) (2020)

## Contents

- [`transformer-pen-and-paper/`](transformer-pen-and-paper/) — a fully worked, step-by-step numerical example of the encoder–decoder Transformer (embeddings, positional encoding, multi-head self-attention, Add & Norm, feed-forward, masked self-attention, cross-attention, output softmax, loss and inference) with a 5-token vocabulary and the sentence “Socrates was most prominent philosopher”.
- [`bert-pen-and-paper/`](bert-pen-and-paper/): step-by-step numerical examples of BERT's two pre-training tasks, [masked language modelling](bert-pen-and-paper/MLM.md) and [next sentence prediction](bert-pen-and-paper/NSP.md). Each goes from the input tokens through the embeddings, the encoder layer and the task head to the loss.
- [`paper-summaries/`](paper-summaries/): detailed, intuition-first summaries of [Attention Is All You Need](paper-summaries/01-attention-is-all-you-need/), [BERT](paper-summaries/02-bert/) and [ViT](paper-summaries/03-vit/), explaining the maths of each paper step by step, each with an exam-style question-and-answer document.
