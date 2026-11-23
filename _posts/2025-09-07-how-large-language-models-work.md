---
layout: post
title: "How Large Language Models Work: Tokens, Prediction, and Post-Training"
date: 2025-09-07 09:00:00-0600
description: A clear, technically careful tour of how a large language model turns text into probabilities and back into text, from tokenization and the training objective to sampling, post-training, and why generation is expensive, with small tested code examples.
tags: llm language-models tokenization tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

A large language model, or LLM, is a neural network that has been trained to predict what comes next in a piece of text. That description sounds modest, but almost everything these systems do, from answering questions to writing code, comes out of that one prediction task repeated many times, together with a surprising amount of engineering around it.

This post walks through the pipeline from raw text to generated text. We look at how text is cut into tokens, what the network outputs, how it is trained, how a base model becomes an assistant, how the next token is chosen, and why generation costs what it does. The aim is an accurate mental model for someone starting in the field, limited to what is publicly documented. Several small code examples are included. All of them are plain Python and NumPy, and every number shown was produced by running the code.

Two related posts go further into specifics: [building a transformer from scratch]({% post_url 2025-06-22-transformer-from-scratch %}) and [speculative decoding]({% post_url 2026-06-14-speculative-decoding %}), a method for faster inference. If backpropagation is unfamiliar, [the earlier post on a neural network from scratch]({% post_url 2024-11-10-neural-network-from-scratch %}) is a good place to start.

## The task: predict the next token

Write a text as a sequence of tokens $$x_1, x_2, \dots, x_T$$. A language model assigns a probability to the whole sequence by breaking it into one-step predictions, using the chain rule of probability:

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1}).
$$

This is an identity, not an assumption. Any distribution over sequences can be written this way. The modeling choice is to represent each factor $$p(x_t \mid x_{<t})$$ with a single neural network that takes the preceding tokens as input and outputs a probability for every token in the vocabulary.

Once we have such a model, generating text is simple. Feed in a prompt, read off the distribution for the next token, choose one token from it, append it to the input, and repeat. The model does not write out a plan before answering. It computes one distribution at a time, and each choice becomes part of the context for the next one. (Any planning it does is implicit in its internal activations.) This is called **autoregressive** generation.

One consequence is easy to miss. The network itself is deterministic: the same input tokens produce the same output distribution, up to small floating-point differences between hardware and batch sizes. Any randomness in a chatbot's answers comes from the step where a token is *sampled* from that distribution, which we return to below.

## Tokens

Neural networks work with numbers, so text must first be converted into integers. The unit of conversion is the token. Using whole words would give a huge vocabulary and no way to handle a word never seen before. Using single characters gives a tiny vocabulary but very long sequences, and a model has to spend effort learning how characters combine into words. Modern models use a compromise, **subword tokens**: common words are single tokens, and rarer words are assembled from pieces.

The most common way to build such a vocabulary is byte-pair encoding, or BPE, which was adapted to neural language processing by Sennrich, Haddow, and Birch [[4]](#references). The training procedure is short to describe:

1. Start with every word split into characters, plus an end-of-word marker.
2. Count how often each adjacent pair of symbols occurs across the corpus.
3. Merge the most frequent pair into a new symbol, and record the merge.
4. Repeat until the vocabulary reaches the desired size.

To encode new text, apply the recorded merges in the order they were learned. Here is a minimal implementation. It follows the small example in the original paper, with a corpus of four words and their counts.

```python
from collections import Counter

def pair_counts(vocab):
    pairs = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, vocab):
    merged = {}
    for symbols, freq in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        merged[tuple(out)] = freq
    return merged

def train_bpe(word_freqs, num_merges):
    vocab = {tuple(word) + ("</w>",): f for word, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)          # most frequent pair (first one wins ties)
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges

def bpe_encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:                           # replay merges in the order they were learned
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = train_bpe(corpus, num_merges=10)
for i, m in enumerate(merges, 1):
    print(f"merge {i:2d}: {m[0]!r} + {m[1]!r}")
for w in ["lowest", "newer", "wider"]:
    print(w, "->", bpe_encode(w, merges))
```

```text
merge  1: 'e' + 's'
merge  2: 'es' + 't'
merge  3: 'est' + '</w>'
merge  4: 'l' + 'o'
merge  5: 'lo' + 'w'
merge  6: 'n' + 'e'
merge  7: 'ne' + 'w'
merge  8: 'new' + 'est</w>'
merge  9: 'low' + '</w>'
merge 10: 'w' + 'i'
lowest -> ['low', 'est</w>']
newer -> ['new', 'e', 'r', '</w>']
wider -> ['wi', 'd', 'e', 'r', '</w>']
```

The first merges are `e`+`s` and then `es`+`t`, which builds the common ending `est` shared by "newest" and "widest". Later merges assemble `low` and `new`. The word "lowest" never appeared in the corpus, but it encodes cleanly as `low` and `est</w>`, two pieces the tokenizer knows. A word like "wider" is partly familiar and partly not, so it ends up as `wi` followed by single characters. The tokenizer is rarely at a loss, because in the worst case it falls back to individual characters, as long as every character is in its vocabulary. Byte-level BPE, described next, removes even that condition.

Production tokenizers use the same idea at scale. GPT-2, for example, applied BPE to raw bytes rather than characters, which guarantees that any string can be encoded, and ended up with a vocabulary of 50,257 tokens [[2]](#references). Other models use different vocabularies of different sizes, so token counts are not comparable between families.

Two practical effects follow from tokenization. First, the model sees tokens, not letters, so tasks that depend on the spelling of a word, such as counting characters, are harder than they look. Second, text in languages or domains that are rare in the tokenizer's training data is split into more pieces, which makes it more expensive to process and leaves less room in the context window.

## From tokens to probabilities

Each token id is mapped to a vector by a learned embedding table. The sequence of vectors passes through a stack of transformer layers [[1]](#references), which mix information across positions using attention and transform it at each position with small feed-forward networks. We build one in code in [the transformer post]({% post_url 2025-06-22-transformer-from-scratch %}). At the end, the vector at each position is projected to one number per vocabulary entry; during generation only the last position's output is used. These numbers are called **logits**, and a softmax turns them into probabilities that are positive and sum to one.

The model only sees a limited number of tokens at once, its **context window**. Anything outside the window does not exist as far as the model is concerned. Text inside the window can be instructions, examples, documents, or earlier turns of a conversation, and the model treats all of it uniformly as a sequence of tokens to continue. This is why prompts work at all, and why giving a model a few examples in the prompt can change its behavior without any change to its weights, a phenomenon documented by Brown et al. [[3]](#references).

## Training: minimizing cross-entropy

Training takes enormous quantities of text, runs it through the model, and adjusts the parameters to make the observed next tokens more probable. At every position, the loss is the negative log of the probability that the model assigned to the token that actually came next. Averaged over positions, this is the cross-entropy, the same quantity used for classification in the [earlier post on neural networks]({% post_url 2024-11-10-neural-network-from-scratch %}), with the vocabulary as the set of classes. Because the training data supplies its own labels, in the form of the text that follows, no human annotation is needed at this stage. This is usually called **self-supervised** learning, and it is what allows pretraining on such large corpora.

The loss is often reported as **perplexity**, the exponential of the average cross-entropy measured in nats. A perplexity of $$k$$ means the model is, on average, as uncertain as if it were choosing uniformly among $$k$$ options at each step.

```python
import numpy as np

def perplexity(token_probs):
    """exp of the average negative log-probability the model gave to the true tokens."""
    return float(np.exp(-np.mean(np.log(token_probs))))

print("uniform over 50,000 tokens:", perplexity([1 / 50_000] * 10))
print("true token gets p=0.5      :", perplexity([0.5] * 10))
```

```text
uniform over 50,000 tokens: 50000.000000000095
true token gets p=0.5      : 2.0
```

A model that knows nothing and spreads its probability evenly over a 50,000-token vocabulary has a perplexity of exactly 50,000. A model that gives the correct token probability 0.5 every time has a perplexity of 2. Perplexity depends on the tokenizer, because a different way of cutting text changes how much each prediction has to carry, so perplexities of models with different vocabularies should not be compared directly.

### Scale

A striking empirical finding is that the loss of a language model on held-out text falls in a smooth, predictable way, close to a power law, as one increases the number of parameters, the amount of training data, or the compute used, provided the other two are not the bottleneck [[5]](#references). Later work refined this: for a fixed training budget, parameters and training tokens should be increased together, with a rule of thumb of roughly 20 training tokens per parameter [[6]](#references). Models in practice are often trained on far more data than that ratio suggests, because a smaller model trained for longer is cheaper to run once it is deployed. These are empirical regularities from experiments on particular model families, not theorems, and the details depend on the data and the architecture.

## From a base model to an assistant

A model trained only on next-token prediction is called a **base model**. Given a question, it does not necessarily answer: the most likely continuation of a question on the internet may be another question, or a forum reply, or something unrelated. Turning a base model into a useful assistant takes further training, often called **post-training**. The details of what is done to commercial systems are generally not public, but the main published techniques are these.

**Supervised fine-tuning.** The model is trained further, with the same next-token loss, on demonstrations of the desired behavior: prompts paired with good responses written by people or selected from other sources. This teaches the format of an assistant's answer and the expected style.

**Learning from preferences.** It is often easier for a person to say which of two responses is better than to write a perfect response. In reinforcement learning from human feedback, usually shortened to RLHF, a *reward model* is trained to predict these human preferences. The language model is then optimized with reinforcement learning to produce responses that score highly, with a penalty that keeps it close to the supervised model so that it does not drift into degenerate text [[7]](#references) [[13]](#references).

**Direct preference optimization.** DPO [[8]](#references) reaches a similar goal without a separate reward model or reinforcement learning loop. Given a prompt $$x$$, a preferred response $$y_w$$, and a rejected response $$y_l$$, it minimizes the following loss, averaged over a dataset of such preference pairs:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\!\left( \beta \left[ \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right] \right),
$$

where $$\pi_\theta$$ is the model being trained, $$\pi_{\text{ref}}$$ is a frozen reference copy, $$\sigma$$ is the logistic function, and $$\beta$$ controls how far the model may move from the reference. In words, the loss rewards raising the probability of the preferred response, relative to the reference model, more than that of the rejected one.

A chat is still a single sequence of tokens. Special tokens mark where the system instructions, the user's turn, and the assistant's turn begin and end, and the model learns to continue after the assistant marker. Nothing about the architecture changes between a base model and a chat model; what changes is the training data and the objective.

## Choosing the next token

At generation time the model outputs logits, and we must decide how to turn them into a token. The simplest rule is **greedy decoding**: always take the most probable token. It is deterministic, but it tends to produce repetitive text, and the single most likely token at each step does not give the most likely sequence overall. The alternative is to sample from the distribution, with some controls that reshape it first.

- **Temperature** divides the logits by a number $$T$$ before the softmax. Values below 1 sharpen the distribution toward the top tokens; values above 1 flatten it. As $$T \to 0$$ sampling approaches greedy decoding, and as $$T \to \infty$$ it approaches a uniform draw.
- **Top-$$k$$** keeps only the $$k$$ highest-scoring tokens and renormalizes [[14]](#references).
- **Top-$$p$$**, or nucleus sampling, keeps the smallest set of tokens whose total probability reaches $$p$$, so the number of candidates adapts to how confident the model is [[9]](#references).

```python
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def token_distribution(logits, temperature=1.0, top_k=None, top_p=None):
    z = np.asarray(logits, dtype=float) / temperature
    if top_k is not None:
        cutoff = np.sort(z)[-top_k]
        z = np.where(z < cutoff, -np.inf, z)
    p = softmax(z)
    if top_p is not None:
        order = np.argsort(-p)
        before = np.cumsum(p[order]) - p[order]       # probability mass ranked above each token
        keep = order[before < top_p]                  # smallest set whose mass reaches top_p
        mask = np.zeros(len(p), dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p = p / p.sum()
    return p

def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

words = ["the", "a", "cat", "dog", "quantum", "banana"]
logits = np.array([3.0, 2.2, 1.5, 1.2, -0.5, -1.0])
settings = [("T=1.0", dict()), ("T=0.5", dict(temperature=0.5)), ("T=2.0", dict(temperature=2.0)),
            ("top-k=2", dict(top_k=2)), ("top-p=0.9", dict(top_p=0.9))]
print(f"{'':10s}" + "".join(f"{w:>9s}" for w in words) + "   entropy (bits)")
for name, kw in settings:
    p = token_distribution(logits, **kw)
    print(f"{name:10s}" + "".join(f"{x:9.3f}" for x in p) + f"   {entropy_bits(p):.2f}")
```

```text
                the        a      cat      dog  quantum   banana   entropy (bits)
T=1.0         0.530    0.238    0.118    0.088    0.016    0.010   1.81
T=0.5         0.781    0.158    0.039    0.021    0.001    0.000   1.01
T=2.0         0.350    0.235    0.165    0.142    0.061    0.047   2.30
top-k=2       0.690    0.310    0.000    0.000    0.000    0.000   0.89
top-p=0.9     0.544    0.244    0.121    0.090    0.000    0.000   1.66
```

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 5, figsize=(14, 3.0), sharey=True)
for ax, (name, kw) in zip(axes, settings):
    p = token_distribution(logits, **kw)
    ax.bar(range(len(words)), p, color="#08519c")
    ax.set_xticks(range(len(words))); ax.set_xticklabels(words, rotation=60, ha="right")
    ax.set_title(f"{name}\nentropy {entropy_bits(p):.2f} bits", fontsize=10)
axes[0].set_ylabel("probability")
fig.tight_layout(); fig.savefig("figures/llm_sampling.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/llm/llm_sampling.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The same six logits turned into probabilities under five settings. Lower temperature concentrates the probability on the top word, higher temperature spreads it out, and top-k and top-p remove the unlikely words and renormalize." %}

The table shows a toy six-word vocabulary with fixed logits. At $$T = 0.5$$ the top word gets 78% of the probability and the entropy drops by almost half, from 1.81 bits to 1.01. At $$T = 2$$ the distribution flattens and the entropy rises to 2.30 bits. Top-$$k$$ with $$k = 2$$ removes everything but "the" and "a". Top-$$p$$ with $$p = 0.9$$ keeps four words, and drops the two unlikely ones, "quantum" and "banana". Note that the top-$$p$$ cutoff is on cumulative probability, so on a different input where the model is more confident it would keep fewer tokens.

## Why generation is expensive

Processing a prompt and generating a reply are different workloads. In the **prefill** phase, the model reads the whole prompt in parallel, which uses the hardware efficiently. In the **decode** phase, it produces one token at a time, and each step depends on the one before. Decoding is typically the slow part.

To avoid recomputing everything for each new token, implementations keep a **key-value cache**. In attention, every earlier token contributes a key vector and a value vector at every layer, and these do not change once computed. Storing them means each decoding step only has to compute the query, key, and value for the newest token, which then attends to the cached keys and values. The price is memory, which grows linearly with the length of the context and with the number of sequences served at once.

```python
def kv_cache_bytes(layers, kv_heads, head_dim, tokens, bytes_per_value=2, batch=1):
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per_value * batch   # 2 = keys and values

GiB = 2 ** 30
per_token = kv_cache_bytes(32, 32, 128, 1)
print(f"7B-class model, 32 KV heads: {per_token/2**20:.2f} MiB per token, "
      f"{kv_cache_bytes(32, 32, 128, 4096)/GiB:.1f} GiB for 4,096 tokens")
print(f"same model, 8 KV heads    : {kv_cache_bytes(32, 8, 128, 1)/2**10:.0f} KiB per token, "
      f"{kv_cache_bytes(32, 8, 128, 8192)/GiB:.1f} GiB for 8,192 tokens")

params, bytes_per_param = 6.74e9, 2           # a 7B-class model stored in 16-bit precision
weight_bytes = params * bytes_per_param
bandwidth, peak_flops = 2.0e12, 312e12        # roughly an A100-class GPU: bytes/s and FLOP/s
print(f"weights: {weight_bytes/1e9:.1f} GB -> at least {weight_bytes/bandwidth*1e3:.1f} ms per token "
      f"(about {bandwidth/weight_bytes:.0f} tokens/s at batch size 1)")
print(f"FLOPs per byte at batch size 1: {2*params/weight_bytes:.0f}; "
      f"the GPU can do {peak_flops/bandwidth:.0f} FLOPs per byte moved")
```

```text
7B-class model, 32 KV heads: 0.50 MiB per token, 2.0 GiB for 4,096 tokens
same model, 8 KV heads    : 128 KiB per token, 1.0 GiB for 8,192 tokens
weights: 13.5 GB -> at least 6.7 ms per token (about 148 tokens/s at batch size 1)
FLOPs per byte at batch size 1: 1; the GPU can do 156 FLOPs per byte moved
```

The cache grows in a straight line with the context, so it is easy to plot. The next figure shows it for one sequence, for the two key-value head counts above, next to the memory taken by the model's weights.

```python
GiB = 2 ** 30
tokens = np.linspace(0, 32768, 200)
fig, ax = plt.subplots(figsize=(7.5, 3.8))
for heads, label, color in ((32, "32 key-value heads (standard multi-head attention)", "#d95f02"),
                            (8, "8 key-value heads (grouped-query attention)", "#08519c")):
    ax.plot(tokens, [kv_cache_bytes(32, heads, 128, t) / GiB for t in tokens], color=color, lw=2, label=label)
ax.axhline(weight_bytes / GiB, color="gray", ls="--", lw=1)
ax.text(500, weight_bytes / GiB + 0.4, "the model's weights (7B parameters, 16-bit)", color="gray", fontsize=9)
ax.set_xlabel("tokens in the context"); ax.set_ylabel("memory (GiB), one sequence"); ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig("figures/llm_kv_cache.png", dpi=170, bbox_inches="tight")
print(f"With 32 key-value heads, the cache of ONE sequence equals the weights at about {weight_bytes / per_token:,.0f} tokens.")
print(f"With 8 key-value heads, that happens at about {weight_bytes / kv_cache_bytes(32, 8, 128, 1):,.0f} tokens.")
```

```text
With 32 key-value heads, the cache of ONE sequence equals the weights at about 25,711 tokens.
With 8 key-value heads, that happens at about 102,844 tokens.
```

{% include figure.liquid path="assets/img/blog/llm/llm_kv_cache.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Key-value cache memory for a single sequence as the context grows, for the 7B-class dimensions used in the text. The dashed line is the memory of the weights themselves." %}

For a single sequence of about 26,000 tokens, the cache of the standard multi-head model is as large as the weights. A server that handles many sequences at once pays that price for each of them, which is why long contexts and large batches are limited by memory long before they are limited by arithmetic.

These numbers describe a model with the dimensions of Llama 2 7B [[12]](#references): 32 layers, 32 attention heads of size 128, 16-bit values. (Llama 2 7B uses standard multi-head attention, so the 8-head variant below is a hypothetical for comparison.) Each token adds half a mebibyte to the cache, so a 4,096-token context takes 2 GiB for a single sequence, and a server handling many sequences needs that much for each of them. Sharing key and value heads across groups of query heads, as in grouped-query attention [[10]](#references), reduces the cache. With 8 key-value heads instead of 32, the cost per token would fall by a factor of four.

The last two lines explain why decoding is limited by memory, not by arithmetic. Each token requires reading every weight once, about 13.5 GB for a 7B-parameter model in 16-bit precision, and on a GPU with roughly 2 TB/s of memory bandwidth that takes at least 6.7 ms. Meanwhile the arithmetic per byte is tiny, about one operation per byte read, against a hardware capability of around 156. These figures are back-of-envelope numbers for an A100-class chip and ignore batching, overheads, and the cache itself. They are meant to show the shape of the bottleneck, which is the starting point for speculative decoding.

## What this picture explains

Several familiar behaviors follow from the pipeline above.

- **Fluent errors.** The training objective rewards putting high probability on text that is a plausible continuation of its context. It has no separate mechanism that checks a statement against the world. When the model lacks the right information, the most plausible continuation can still be a confident and wrong one, which is usually called hallucination. Post-training and retrieval can reduce this, but nothing in the basic design removes it.
- **Sensitivity to wording.** The output is a function of the exact token sequence. Rephrasing a prompt changes the input and therefore the distribution.
- **Knowledge cutoffs.** What the model knows is whatever was in its training data, plus whatever is placed in its context. Giving it relevant documents at query time, an approach known as retrieval-augmented generation [[11]](#references), is a standard way to supply fresher or private information.
- **Limits of the window.** Anything outside the context window is invisible, and very long contexts are expensive because of the cache and attention costs.

## Where efficiency research goes

Because the costs above are so large, a good part of current research aims at making models cheaper to run. A short map of the main directions: **quantization** stores weights, and sometimes the cache, with fewer bits so that less data moves per token. **Pruning and distillation** make the model itself smaller, the second by training a small model to imitate a large one. **Mixture-of-experts** designs activate only part of the network for each token. **Cache compression and attention variants** reduce the memory that long contexts consume. And **speculative decoding**, the subject of [the speculative decoding post]({% post_url 2026-06-14-speculative-decoding %}), reduces the number of expensive sequential steps without changing the output distribution at all.

## References

1. A. Vaswani et al. "Attention Is All You Need." *NeurIPS*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. "Language Models are Unsupervised Multitask Learners." OpenAI technical report, 2019.
3. T. B. Brown et al. "Language Models are Few-Shot Learners." *NeurIPS*, 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
4. R. Sennrich, B. Haddow, and A. Birch. "Neural Machine Translation of Rare Words with Subword Units." *ACL*, 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
5. J. Kaplan et al. "Scaling Laws for Neural Language Models." 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
6. J. Hoffmann et al. "Training Compute-Optimal Large Language Models." 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
7. L. Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*, 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
8. R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." *NeurIPS*, 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
9. A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi. "The Curious Case of Neural Text Degeneration." *ICLR*, 2020. [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
10. J. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *EMNLP*, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
11. P. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
12. H. Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." 2023. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
13. P. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei. "Deep Reinforcement Learning from Human Preferences." *NeurIPS*, 2017. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
14. A. Fan, M. Lewis, and Y. Dauphin. "Hierarchical Neural Story Generation." *ACL*, 2018. [arXiv:1805.04833](https://arxiv.org/abs/1805.04833)
