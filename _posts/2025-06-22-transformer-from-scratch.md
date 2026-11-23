---
layout: post
title: "A Transformer from Scratch: Attention, Residual Blocks, and Next-Token Prediction"
date: 2025-06-22 09:00:00-0600
description: Build a small GPT-style transformer in PyTorch, with attention, layer normalization, and residual connections written out explicitly, then verify it with causality and reference checks, train it on character-level text, and sample from it.
tags: transformer attention pytorch language-models tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

The transformer [[1]](#references) is the architecture behind nearly every large language model, and its core is short. A working version fits in about a hundred lines. The difficulty is not the amount of code but the number of small decisions packed into it: which dimension to normalize over, where the mask goes, why the scores are scaled, why the residual stream looks the way it does. Each is easy to get subtly wrong and hard to notice from the training loss alone.

In this post we build a small decoder-only transformer, the GPT-style model used for text generation [[2]](#references), and train it to predict the next character on a small public dataset. We use PyTorch for tensors and automatic differentiation, but we write the attention, the layer structure, and the model ourselves instead of calling `nn.TransformerDecoder` or `nn.MultiheadAttention`. Along the way we derive the pieces, check the implementation against an independent reference, and verify that the model cannot see the future. If you want the background on what a language model is and how it is used, [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}) covers it. For backpropagation itself, see [the post on a neural network from scratch]({% post_url 2024-11-10-neural-network-from-scratch %}).

The code is one script, shown in sections. Copy the blocks in order into a file such as `gpt_from_scratch.py` and run `python gpt_from_scratch.py`. The results below were produced with Python 3.10.10 and PyTorch 2.14.1 on a laptop CPU, and the full run takes about nine minutes there.

```bash
pip install torch
```

PyTorch is deterministic given a seed on one machine, but results can differ in the last digits across hardware and library versions, so expect numbers close to the ones below and not necessarily identical. The data download uses Python's standard library; if you see an SSL certificate error on macOS with a python.org installation, running the "Install Certificates.command" file that ships with Python fixes it.

## The task and the data

We train on the "Tiny Shakespeare" text file, about 1.1 million characters of Shakespeare plays, which Andrej Karpathy assembled for his character-level language modeling demonstrations [[8]](#references). The model works at the level of **characters**: its vocabulary is the set of distinct characters in the file, 65 of them. This keeps the tokenizer trivial so that attention and the transformer are the only moving parts. Real language models use subword tokens, as described in [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}).

The model is trained to predict, at every position, the character that comes next. If the text is $$x_1, \dots, x_N$$, we take a window of $$T$$ consecutive characters as input and the same window shifted by one as the target, so that the target at position $$t$$ is $$x_{t+1}$$. Because of the causal mask that we add below, a window of length $$T$$ gives us $$T$$ training examples at once, each predicting from a different amount of context.

We hold out the last 10% of the text for validation. The split is contiguous and not random, since neighboring windows overlap heavily and a random split would leak training text into the validation set.

```python
import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = "cpu"

# ---------- data ----------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(URL, "input.txt")
text = open("input.txt", encoding="utf-8").read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))                    # contiguous split, not a shuffle
train_data, val_data = data[:n_train], data[n_train:]
print(f"{len(text):,} characters, vocabulary size {vocab_size}")
print(f"train {len(train_data):,} tokens, validation {len(val_data):,} tokens")

block_size, batch_size = 128, 32
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])          # inputs   (B, T)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # targets  (B, T), shifted by one
    return x.to(device), y.to(device)
```

```text
1,115,394 characters, vocabulary size 65
train 1,003,854 tokens, validation 111,540 tokens
```

## Shapes

Keeping track of tensor shapes is most of the work in implementing a transformer, so let us fix notation. We write $$B$$ for the batch size, $$T$$ for the sequence length (the block size, 128 here), $$C$$ for the embedding width, $$H$$ for the number of attention heads, $$d_h = C / H$$ for the width of one head, and $$V$$ for the vocabulary size.

| Tensor | Shape |
|---|---|
| token ids `idx`, targets | $$(B, T)$$ |
| embeddings, residual stream $$x$$ | $$(B, T, C)$$ |
| queries, keys, values, per head | $$(B, H, T, d_h)$$ |
| attention scores and weights | $$(B, H, T, T)$$ |
| logits | $$(B, T, V)$$ |

Almost every bug in a first implementation is a shape that happens to be valid but means the wrong thing, such as merging dimensions in the wrong order. We will add checks for exactly that.

## Self-attention

The central operation lets each position gather information from the other positions. Every position emits three vectors computed from its input $$x_t \in \mathbb{R}^C$$ by learned linear maps: a **query** $$q_t$$, a **key** $$k_t$$, and a **value** $$v_t$$. Position $$t$$ compares its query with the keys of all positions, converts the comparison scores to weights with a softmax, and takes the weighted average of the values:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^{\top}}{\sqrt{d_k}} + M \right) V.
$$

Here $$Q$$, $$K$$, $$V$$ stack the queries, keys, and values of all positions as rows. The matrix $$Q K^\top$$ has entry $$(s, t)$$ equal to $$q_s \cdot k_t$$, the score between query position $$s$$ and key position $$t$$, so its shape is $$T \times T$$.

**The mask.** In a language model, position $$s$$ must not use information from positions after it, since those are the characters we are trying to predict. The matrix $$M$$ is $$0$$ where $$t \le s$$ and $$-\infty$$ where $$t > s$$. Adding $$-\infty$$ before the softmax sets the corresponding weights to exactly zero. Each row of weights still sums to one over the allowed positions. This is the **causal mask**, and it is what allows all $$T$$ positions to be trained in parallel while each one only sees its past.

**The scale.** The scores are divided by $$\sqrt{d_k}$$, where $$d_k$$ is the width of the keys. The reason [[1]](#references) is a variance argument. If the entries of $$q$$ and $$k$$ are independent with zero mean and unit variance, then $$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$ has mean zero and variance $$d_k$$. Without scaling, the typical size of the scores would grow with the width of the head, the softmax would put almost all of its weight on one position, and its gradients would become very small. Dividing by $$\sqrt{d_k}$$ restores unit variance.

**Multiple heads.** Instead of one attention operation with width $$C$$, we split the width into $$H$$ heads of size $$d_h = C/H$$, run attention independently in each, concatenate the results, and apply a final linear map $$W_O$$. The total cost is about the same as one wide head, and each head is free to learn a different pattern of where to look. In the code, one linear layer produces queries, keys, and values for all heads at once, and a reshape separates the heads.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)       # queries, keys, values in one matmul
        self.proj = nn.Linear(n_embd, n_embd)          # output projection W_O
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)             # True where attention is allowed

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        H, dh = self.n_head, C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)                          # each (B, T, C)
        q = q.view(B, T, H, dh).transpose(1, 2)                        # (B, H, T, dh)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)             # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf")) # no looking ahead
        weights = F.softmax(scores, dim=-1)                            # each row sums to 1
        y = self.attn_drop(weights) @ v                                # (B, H, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)               # concatenate heads
        y = self.resid_drop(self.proj(y))
        return (y, weights) if return_weights else y
```

Read the shapes along the forward pass. After `qkv`, each of `q`, `k`, `v` is $$(B, T, C)$$. The `view` followed by `transpose(1, 2)` turns them into $$(B, H, T, d_h)$$, so that the matrix product `q @ k.transpose(-2, -1)` runs over the last two dimensions and produces $$(B, H, T, T)$$. The final `transpose` and `view` concatenate the heads back into $$(B, T, C)$$. The order matters: the reshape to separate heads must be `view(B, T, H, dh)` and then a transpose. Writing `view(B, H, T, dh)` directly gives the same shape but silently mixes positions into heads.

## The rest of the block

Attention moves information *between* positions. The other half of a transformer block processes information *within* each position, using a two-layer network with the same weights at every position:

$$
\text{MLP}(x) = W_2\, \text{GELU}(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{4C \times C},\; W_2 \in \mathbb{R}^{C \times 4C}.
$$

The hidden layer is four times wider than the residual stream, a convention from the original paper, and GELU [[5]](#references) is a smooth variant of the ReLU nonlinearity.

The two sublayers are wrapped with **residual connections** [[4]](#references) and **layer normalization** [[3]](#references). Layer normalization rescales each position's vector to zero mean and unit variance across its $$C$$ features, then applies a learned scale and shift. We use the "pre-norm" arrangement used by GPT-2, in which normalization is applied to the input of each sublayer and the sublayer's output is added back to the unnormalized stream:

$$
x \leftarrow x + \text{Attn}(\text{LN}(x)), \qquad x \leftarrow x + \text{MLP}(\text{LN}(x)).
$$

It is useful to think of $$x$$ as a *residual stream* that runs through the whole network, from the embeddings to the output. Each sublayer reads from it and adds a contribution to it. Because the identity path is never interrupted, gradients flow back through all layers easily. The original Transformer [[1]](#references) normalized after the addition instead; later analysis showed that the pre-norm arrangement trains more stably, especially in deep networks [[6]](#references).

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # residual connection around attention
        x = x + self.mlp(self.ln2(x))     # residual connection around the MLP
        return x
```

## The full model

The model embeds the input characters, adds position information, applies a stack of blocks, normalizes, and maps back to the vocabulary.

**Positions.** Without a mask, attention has no notion of order: permuting the input positions would simply permute the outputs. The causal mask does reveal some order implicitly, since position $$s$$ can only see positions up to $$s$$, but that is a weak signal, and models are given explicit position information. Here we add a learned vector for each position, a table of size $$T \times C$$ indexed by position. Other choices are common. The original paper used fixed sinusoids, and many recent models rotate the queries and keys by a position-dependent angle, called rotary embeddings [[9]](#references).

**Weight tying.** The final linear layer that produces logits has a weight matrix of shape $$V \times C$$, the same shape as the token embedding table. We share one matrix between the two [[7]](#references), which saves parameters and often helps.

**Initialization.** Weights are drawn from a normal distribution with standard deviation 0.02. The output projections of the attention and MLP sublayers, the matrices that write into the residual stream, are scaled down by $$1/\sqrt{2L}$$, where $$L$$ is the number of layers. This is the GPT-2 convention [[2]](#references), and it keeps the variance of the residual stream from growing with depth. A useful sanity check follows from it. At initialization, the model should know nothing, so its output distribution should be close to uniform and its loss close to $$\ln V = \ln 65 \approx 4.17$$.

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight       # tie input and output embeddings
        self.apply(self._init)
        for name, p in self.named_parameters():         # scale residual-branch outputs
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                     # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])         # crop to the context window
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = GPT(vocab_size, block_size, n_embd=128, n_head=4, n_layer=4).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
```

```text
parameters: 818,048
```

We can verify the parameter count by hand, which is a good habit. One block has $$3C^2 + 3C$$ parameters in the query-key-value layer, $$C^2 + C$$ in the output projection, $$8C^2 + 5C$$ in the MLP, and $$4C$$ in its two layer norms, for $$12C^2 + 13C$$ in total. With $$C = 128$$ and four blocks, that is $$4 \times 198{,}272 = 793{,}088$$. Add the token embedding $$65 \times 128 = 8{,}320$$, the position table $$128 \times 128 = 16{,}384$$, and the final layer norm's $$256$$ parameters, and the total is $$818{,}048$$. This matches the number printed above. The output layer adds nothing, since its weight is tied to the token embedding.

## Checking the implementation

Before training, it is worth testing the three properties that we most care about. These tests are cheap and catch the most common errors.

1. **The loss at initialization should be close to $$\ln V$$.**
2. **Causality.** Changing the input token at position 50 must not change the logits at any earlier position, and it should change the logits from position 50 onward.
3. **Attention weights.** Each row should sum to one, with exactly zero weight above the diagonal. And our attention should agree with PyTorch's reference `scaled_dot_product_attention` with `is_causal=True`.

```python
model.eval()
x, y = get_batch("train")
with torch.no_grad():
    _, loss = model(x, y)
print(f"initial loss {loss.item():.3f}   ln(vocab) = {math.log(vocab_size):.3f}")

# 1. Causality: changing token t must not change the logits at positions < t.
with torch.no_grad():
    a, _ = model(x)
    x2 = x.clone(); x2[:, 50] = (x2[:, 50] + 1) % vocab_size
    b, _ = model(x2)
print("max logit change before position 50:", (a[:, :50] - b[:, :50]).abs().max().item())
print("max logit change at/after position 50:", (a[:, 50:] - b[:, 50:]).abs().max().item())

# 2. Attention weights: rows sum to one, nothing above the diagonal.
attn = model.blocks[0].attn
with torch.no_grad():
    h = model.tok_emb(x) + model.pos_emb(torch.arange(block_size))
    _, w = attn(model.blocks[0].ln1(h), return_weights=True)
print("row sums min/max:", w.sum(-1).min().item(), w.sum(-1).max().item())
print("max weight above diagonal:", torch.triu(w, diagonal=1).max().item())

# 3. Agreement with PyTorch's reference implementation.
with torch.no_grad():
    B, T, C = h.shape
    H, dh = 4, C // 4
    q, k, v = attn.qkv(model.blocks[0].ln1(h)).split(C, dim=2)
    q, k, v = [t.view(B, T, H, dh).transpose(1, 2) for t in (q, k, v)]
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mine = w @ v
print("max difference from F.scaled_dot_product_attention:", (ref - mine).abs().max().item())
```

```text
initial loss 4.209   ln(vocab) = 4.174
max logit change before position 50: 0.0
max logit change at/after position 50: 1.464619755744934
row sums min/max: 0.9999997615814209 1.000000238418579
max weight above diagonal: 0.0
max difference from F.scaled_dot_product_attention: 8.195638656616211e-08
```

The initial loss is 4.209, close to $$\ln 65 = 4.174$$, as it should be. The logits before position 50 are unchanged to the last bit, while those after it changed by up to about 1.5, so information flows forward in the sequence and never backward. The attention weights sum to one within floating-point error and are exactly zero above the diagonal, and the output of our attention differs from the reference implementation by about $$8 \times 10^{-8}$$, which is rounding error for 32-bit floats. These tests would catch a missing mask, a softmax over the wrong dimension, a missing scale, or a wrongly merged head dimension.

## Training

The optimizer is AdamW [[10]](#references), which handles weight decay separately from the gradient update. The learning rate warms up linearly for 100 steps and then decays along a cosine curve to one tenth of its peak. Gradients are clipped to norm 1, a standard guard against occasional large updates. Every 500 steps we measure the loss on the training and validation sets with dropout turned off. Calling `model.eval()` before measuring, and `model.train()` afterward, is easy to forget and matters.

```python
max_steps, eval_every, eval_batches = 3000, 500, 50
lr_max, lr_min, warmup = 2e-3, 2e-4, 100

optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)

def lr_at(step):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()                                   # turns dropout off
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_batches)
        for i in range(eval_batches):
            xb, yb = get_batch(split)
            losses[i] = model(xb, yb)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out

evals = []
history = []                                       # training loss at every step, for plotting
start = time.time()
for step in range(max_steps + 1):
    if step % eval_every == 0:
        l = estimate_loss()
        evals.append((step, l["train"], l["val"]))
        print(f"step {step:4d} | train loss {l['train']:.3f} | val loss {l['val']:.3f} | {time.time()-start:5.0f}s")
    if step == max_steps:
        break
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    history.append(loss.item())
```

```text
step    0 | train loss 4.206 | val loss 4.206 |     5s
step  500 | train loss 2.041 | val loss 2.100 |    92s
step 1000 | train loss 1.722 | val loss 1.873 |   178s
step 1500 | train loss 1.560 | val loss 1.742 |   264s
step 2000 | train loss 1.458 | val loss 1.660 |   352s
step 2500 | train loss 1.413 | val loss 1.618 |   438s
step 3000 | train loss 1.398 | val loss 1.594 |   525s
```

The validation loss falls from 4.21, the uniform-guess level, to 1.59 after 3,000 steps. Losses here are in nats per character, so a loss of 1.59 corresponds to a perplexity of about 4.9: on average the model is as uncertain about the next character as if it were choosing among five equally likely ones, compared with 65 at the start. In bits, that is 2.30 bits per character against $$\log_2 65 = 6.02$$ for a uniform guess.

The training loss ends at 1.40, below the validation loss, and the difference of about 0.2 is the generalization gap. It is modest for this model and this amount of training. The validation loss was still decreasing at the end of the run, from 1.62 at step 2,500 to 1.59 at step 3,000, so this model is limited by training time and size more than by overfitting. With a much larger model or many more epochs over the same 1 MB of text, the gap would grow and validation loss would eventually turn upward.

The loss history is recorded in the training loop, and the plotting code is short:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

smooth = np.convolve(history, np.ones(25) / 25, mode="valid")
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(np.arange(len(history)), history, color="#c6dbef", lw=0.8, label="training loss (every step)")
ax.plot(np.arange(len(smooth)) + 12, smooth, color="#08519c", lw=1.8, label="training loss (25-step average)")
steps, tr, va = zip(*evals)
ax.plot(steps, va, "o-", color="#d95f02", lw=1.8, label="validation loss")
ax.axhline(math.log(vocab_size), color="gray", ls="--", lw=1)
ax.text(2000, math.log(vocab_size) + 0.05, "uniform guess, ln 65", color="gray", ha="center")
ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats per character)")
ax.set_ylim(1.2, 4.5); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/transformer_loss.png", dpi=170)
```

{% include figure.liquid path="assets/img/blog/transformer/transformer_loss.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Training and validation loss over 3,000 steps. The pale line is the raw per-step training loss, which is noisy because each step sees only 32 random windows; the dark line averages 25 steps. The dashed line is the loss of a model that guesses uniformly among the 65 characters." %}

The curve is worth a look beyond the final numbers. The loss drops steeply in the first few hundred steps, when the model is learning character frequencies and common letter pairs, and then bends over into a slow decline as it picks up longer patterns such as spelling and the layout of a script. The warmup and cosine schedule show up as a smooth flattening at the end.

## Generating text

To sample, we feed the model a context, take the distribution at the last position, divide the logits by a temperature, sample a character, append it, and repeat. The context is cropped to the last `block_size` characters, because the position table has no entries beyond that.

```python
model.eval()
torch.manual_seed(0)
context = torch.zeros((1, 1), dtype=torch.long)    # start from token 0 (a newline in this dataset)
print(decode(model.generate(context, max_new_tokens=400, temperature=0.8)[0].tolist()))

torch.save(model.state_dict(), "gpt_shakespeare.pt")
```

```text
Thy father had self my Houseep,
And in I the earth of thy hound bear change.

KING RICHARD II:
What a son to him be many soul
And trunk-day for our purposed brief him his gods!

SICINIUS:
Why! no.

SICINIUS:
That
This thou starst you as our defence.

ANGELO:
A must noble in be your founds from my hand.

LORD F WILLOXERD:
If I thou put of some to my rage heart be well,
Or go well will the dext like
```

This is the first sample from a fixed seed, not a selection of the best of several. The model has clearly picked up the surface structure of a play: speaker names in capitals followed by colons, short lines, punctuation in plausible places, and mostly real English words. It has not learned much grammar or meaning, and it occasionally invents words. That is what to expect from fewer than a million parameters trained for nine minutes on a CPU on one megabyte of text. The point of the exercise is that the architecture works and that the checks above give us confidence it is implemented correctly, not that the output is good prose.

## Does the model use its context?

A language model is only useful if the characters far back in the window can help predict the next one. We can test that for our model directly. Take the validation text in non-overlapping windows of 128 characters, and measure the average loss at each position within the window. Position 1 has only a single character of context; position 128 has 127.

```python
# Loss at each position in the context, averaged over validation windows.
model.eval()
n_windows = (len(val_data) - 1) // block_size          # non-overlapping validation windows
starts = torch.arange(n_windows) * block_size
xs = torch.stack([val_data[i : i + block_size] for i in starts])
ys = torch.stack([val_data[i + 1 : i + block_size + 1] for i in starts])
with torch.no_grad():
    logits, _ = model(xs)
    per_token = F.cross_entropy(logits.reshape(-1, vocab_size), ys.reshape(-1), reduction="none").view(n_windows, block_size)
by_position = per_token.mean(0).numpy()
print("mean loss at positions 1, 8, 32, 128:", [round(float(by_position[i - 1]), 2) for i in (1, 8, 32, 128)])

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(np.arange(1, block_size + 1), by_position, color="#08519c", lw=1.8)
ax.set_xlabel("position in the context window"); ax.set_ylabel("validation loss (nats)")
fig.tight_layout(); fig.savefig("figures/transformer_position_loss.png", dpi=170)
```

```text
mean loss at positions 1, 8, 32, 128: [2.56, 1.65, 1.61, 1.63]
```

{% include figure.liquid path="assets/img/blog/transformer/transformer_position_loss.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Average validation loss at each position in the 128-character window, over all non-overlapping validation windows. Each point is an average over several hundred windows, so the remaining wiggle is sampling noise." %}

The first few positions are much harder, with a loss of about 2.6 nats at position 1, and the curve falls to roughly 1.65 by position 8. After that it is flat, at about 1.6 for the rest of the window. So for this small character-level model, nearly all of the benefit of context arrives within the first handful of characters. It uses recent letters to spell words and complete them, and it does not get measurably better at predicting when it has 100 characters of history instead of 20. That is a limit of this model, its size, and its training budget, and not a property of the architecture. Larger models trained on subword tokens do make use of much longer contexts.

## Looking at attention

The last diagnostic is a picture of the attention weights themselves. The next figure feeds the string `ROMEO:\nBut soft, what light` through the trained model and plots the weights of the four heads in the last layer. Row $$s$$ shows where the character at position $$s$$ looks, and each row sums to one over the columns to its left, which is why everything above the diagonal is empty.

```python
def attention_weights(model, idx, layer):
    """Attention weights (H, T, T) of one layer for a single sequence idx of shape (1, T)."""
    with torch.no_grad():
        x = model.tok_emb(idx) + model.pos_emb(torch.arange(idx.size(1)))
        for block in model.blocks[:layer]:
            x = block(x)
        _, w = model.blocks[layer].attn(model.blocks[layer].ln1(x), return_weights=True)
    return w[0]

text_example = "ROMEO:\nBut soft, what light"
idx = torch.tensor([encode(text_example)])
labels = [c if c != "\n" else "\\n" for c in text_example]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for h, ax in enumerate(axes):
    w = attention_weights(model, idx, layer=3)[h].numpy()
    ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"layer 4, head {h + 1}", fontsize=10)
    if h == 0:
        ax.set_ylabel("query (the character doing the looking)")
    ax.set_xlabel("key (the character being looked at)")
fig.tight_layout(); fig.savefig("figures/transformer_attention.png", dpi=170)
```

{% include figure.liquid path="assets/img/blog/transformer/transformer_attention.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Attention weights of the four heads in the final layer, for one short input. Darker means more weight. The triangular shape is the causal mask: a character can only look at itself and earlier characters." %}

Two things stand out. Much of the weight sits on or just below the diagonal, meaning that characters attend heavily to the few characters right before them, which is consistent with the finding above that local context carries most of the signal. And the heads are not copies of each other: some spread weight over several recent characters while others concentrate on one particular earlier position, such as the colon after the speaker's name. It is tempting to give each head a job description from a picture like this, and that temptation should be resisted. A pattern in the weights shows where information could flow, not what the network does with it, and establishing what a head is for takes interventions such as ablating it and measuring the change in loss.

Note that `generate` recomputes the whole window at every step, so the cost of producing each character grows with the context. Production systems avoid this by caching the keys and values of earlier positions, which is the key-value cache discussed in [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}), and which is also what makes [speculative decoding]({% post_url 2026-06-14-speculative-decoding %}) interesting.

## Common implementation errors

- **No mask, or the wrong mask.** The training loss falls suspiciously fast and far, because the model can read the answer from the future, while sampling produces garbage. The causality test above detects it.
- **Targets not shifted.** If the target at position $$t$$ is $$x_t$$ and not $$x_{t+1}$$, the model learns to copy its input and the loss goes toward zero without any useful learning.
- **Softmax over the wrong dimension.** Normalizing over queries instead of keys gives weights that sum to one in the wrong direction. The row-sum check catches this.
- **Forgetting the $$1/\sqrt{d_k}$$ scale.** Training still works at small sizes and degrades as width grows, which makes it hard to notice. Comparing against a reference implementation catches it.
- **Reshaping heads incorrectly.** A `view` that skips the transpose has the right shape and the wrong meaning. Again, the reference comparison is the test.
- **Dropout left on during evaluation.** Validation loss is noisy and too high. Remember `model.eval()`.
- **Positions beyond the block size.** The position table has `block_size` rows, so longer inputs raise an index error, or in other designs silently misbehave. Crop the context when generating.
- **Judging by the training loss alone.** Always track a held-out loss. A model can reduce its training loss by memorizing.

## What real language models change

The model here keeps the structure of the models behind modern LLMs and changes mostly the scale and a handful of design choices. Production models work with subword tokens and vocabularies of tens of thousands of entries instead of 65 characters. Many replace layer normalization with a simpler variant called RMSNorm, replace GELU with a gated activation, use rotary position embeddings [[9]](#references), and share key and value heads between groups of query heads to shrink the cache. They compute attention with fused kernels such as FlashAttention [[11]](#references), which produce the same result without ever storing the full $$T \times T$$ weight matrix, since the cost of attention in time and memory grows quadratically with context length. And they are trained on thousands of accelerators, with mixed-precision arithmetic, over trillions of tokens. None of these changes alter the basic idea: a stack of attention and MLP layers around a residual stream, trained to predict the next token.

## Where to go next

Some natural extensions, each a small change to the script: replace the learned position table with rotary embeddings; add a key-value cache to `generate` and check that it produces the same outputs as the version here; swap characters for a byte-pair-encoding tokenizer; train a larger model for longer and plot how the validation loss changes with parameters, in the spirit of the scaling-law experiments mentioned in [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}).

## References

1. A. Vaswani et al. "Attention Is All You Need." *NeurIPS*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. "Language Models are Unsupervised Multitask Learners." OpenAI technical report, 2019.
3. J. L. Ba, J. R. Kiros, and G. E. Hinton. "Layer Normalization." 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
4. K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning for Image Recognition." *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
5. D. Hendrycks and K. Gimpel. "Gaussian Error Linear Units (GELUs)." 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
6. R. Xiong et al. "On Layer Normalization in the Transformer Architecture." *ICML*, 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
7. O. Press and L. Wolf. "Using the Output Embedding to Improve Language Models." *EACL*, 2017. [arXiv:1608.05859](https://arxiv.org/abs/1608.05859)
8. A. Karpathy. "char-rnn" and the Tiny Shakespeare dataset, 2015. [github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)
9. J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
10. I. Loshchilov and F. Hutter. "Decoupled Weight Decay Regularization." *ICLR*, 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
11. T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
