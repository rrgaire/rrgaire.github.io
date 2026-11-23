---
layout: post
title: "Vision-Language Models from Scratch: Teaching a Language Model to See"
date: 2025-12-14 10:00:00-0600
description: Build a small vision-language model in PyTorch, with a patch-based image encoder, a projector, and a language model, train it to caption and answer questions about images, and test what happens when you throw away visual tokens.
tags: vlm multimodal vision-transformer tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

A language model reads and writes text. A vision-language model, or VLM, also looks at images. Ask it what is in a photo, what a chart says, or whether a sign is in focus, and it answers in words. Over the last few years this has gone from a research curiosity to a standard feature of large models, and the design behind most of them is simpler than it looks.

The central idea is to turn an image into a short sequence of vectors that a language model can read as if they were word embeddings. After that, the language model does what it always does: it predicts the next token. In this post we build a small version of this recipe from scratch and train it on a toy world of colored shapes, where we can check every answer. We then use it for something relevant to the practical cost of VLMs: measuring what happens when we drop most of the visual tokens before the language model sees them.

You should be comfortable with the transformer, which we build in [an earlier post]({% post_url 2025-06-22-transformer-from-scratch %}), and with the idea of a language model, covered in [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}).

## Two families of vision-language models

There are two main ways to connect images and text, and it helps to separate them early.

**Contrastive models** such as CLIP [[1]](#references) train an image encoder and a text encoder together so that a picture and its caption land close to each other in a shared embedding space, and mismatched pairs land far apart. They do not generate text. They are used to score how well an image matches a piece of text, which gives zero-shot classification (compare an image with the text "a photo of a dog" and "a photo of a cat" and pick the closer one) and image search. They are also a common source of the image encoder in the second family.

**Generative models** such as LLaVA [[2]](#references) take images as input and produce text. A vision encoder turns the image into a set of feature vectors, a small **projector** maps those vectors into the language model's embedding space, and the language model reads them as a prefix before the question and then writes the answer. Training uses the usual next-token loss on the answer. Other designs feed the image into the language model through cross-attention layers instead of as a prefix, as in Flamingo [[3]](#references), or compress the image features into a fixed number of learned query vectors first, as in BLIP-2 [[4]](#references).

This post builds the second family in its simplest form, the prefix design, and shows the contrastive loss in a short snippet so that you can see how the two relate.

## A toy world

Real VLMs are trained on hundreds of millions of image-text pairs, which is not something we can do on a laptop. Instead we build a world small enough to train in minutes but structured enough to ask real questions about. Each image is $$48 \times 48$$ pixels with a $$3 \times 3$$ grid of cells. It contains one to three shapes, each of a different kind (circle, square, triangle), each in its own cell, and each in one of four colors. A small random jitter moves each shape inside its cell, so no two images are pixel-identical.

For every image we generate text tasks. A **caption** lists the shapes in reading order. A **count** question asks how many shapes there are. A **color** question asks what color a named shape is. A **where** question asks which of nine cells a named shape is in. An **existence** question asks whether a shape is present, with about half of the answers being no. The scene description, the rendering, and the tasks are all in the code:

```python
import math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

COLORS = {"red": (220, 40, 40), "green": (40, 160, 60), "blue": (50, 90, 220), "yellow": (240, 200, 30)}
SHAPES = ["circle", "square", "triangle"]
CELL_NAMES = ["top left", "top middle", "top right", "middle left", "center", "middle right",
              "bottom left", "bottom middle", "bottom right"]
CELL, SIZE = 16, 48                                        # a 3x3 grid of 16-pixel cells -> a 48x48 image

def shape_mask(kind, size=11):
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    if kind == "circle":
        return (yy - c) ** 2 + (xx - c) ** 2 <= (size / 2) ** 2
    if kind == "square":
        return np.ones((size, size), dtype=bool)
    return np.abs(xx - c) <= yy * c / (size - 1)           # triangle pointing up

MASKS = {k: shape_mask(k) for k in SHAPES}

def make_scene(rng):
    """1-3 shapes of different kinds, each in its own cell of the 3x3 grid, in random colors."""
    n = int(rng.integers(1, 4))
    kinds = rng.permutation(SHAPES)[:n]
    cells = rng.choice(9, size=n, replace=False)
    colors = rng.choice(list(COLORS), size=n)
    return sorted(zip(cells.tolist(), kinds.tolist(), colors.tolist()))     # reading order: row by row

def render(scene, rng):
    img = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    for cell, kind, color in scene:
        r, c = divmod(cell, 3)
        dy, dx = rng.integers(0, 5, size=2)               # small random offset inside the cell
        y0, x0 = r * CELL + dy, c * CELL + dx
        m = MASKS[kind]
        img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m] = COLORS[color]
    return img

def caption(scene):
    parts = [f"a {color} {kind}" for _, kind, color in scene]
    return (" , ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + " ."

rng = np.random.default_rng(0)
demo = [make_scene(rng) for _ in range(3)]
for s in demo:
    print(caption(s))
```

```text
a yellow triangle , a yellow square and a blue circle .
a blue circle and a yellow triangle .
a green triangle .
```

```python
NUMBER_WORDS = ["one", "two", "three"]

def make_examples(scene, rng):
    """Text tasks for one scene, as (task, prompt, answer) triples of space-separated words."""
    kinds = {k: (cell, color) for cell, k, color in scene}
    present = list(kinds)
    absent = [k for k in SHAPES if k not in kinds]
    ex = [("caption", "<cap>", caption(scene)),
          ("count", "how many shapes ?", NUMBER_WORDS[len(scene) - 1])]
    k = present[rng.integers(len(present))]
    ex.append(("color", f"what color is the {k} ?", kinds[k][1]))
    k = present[rng.integers(len(present))]
    ex.append(("where", f"where is the {k} ?", CELL_NAMES[kinds[k][0]]))
    ex.append(("exist", f"is there a {present[rng.integers(len(present))]} ?", "yes"))
    if absent:
        ex.append(("exist", f"is there a {absent[rng.integers(len(absent))]} ?", "no"))
    else:
        ex.append(("exist", f"is there a {present[rng.integers(len(present))]} ?", "yes"))
    return ex

words = sorted({w for k in SHAPES for w in [k]} | set(COLORS) | set(CELL_NAMES[i].split()[j] for i in range(9) for j in range(len(CELL_NAMES[i].split())))
               | set(NUMBER_WORDS) | {"yes", "no", "a", "and", ",", ".", "?", "<cap>", "how", "many", "shapes", "what",
                                       "color", "is", "the", "there", "where"})
vocab = ["<pad>", "<eos>"] + words
stoi = {w: i for i, w in enumerate(vocab)}
print(f"vocabulary: {len(vocab)} words")

TEXT_LEN = 24
def encode_example(prompt, answer):
    p, a = prompt.split(), answer.split()
    ids = [stoi[w] for w in p + a] + [stoi["<eos>"]]
    mask = [0] * len(p) + [1] * (len(a) + 1)              # the loss is computed on the answer only
    pad = TEXT_LEN - len(ids)
    return ids + [0] * pad, mask + [0] * pad

def build_split(n_scenes, seed):
    rng = np.random.default_rng(seed)
    images, ids, masks, tasks, scene_idx = [], [], [], [], []
    for s in range(n_scenes):
        scene = make_scene(rng)
        images.append(render(scene, rng))
        for task, prompt, answer in make_examples(scene, rng):
            i, m = encode_example(prompt, answer)
            ids.append(i); masks.append(m); tasks.append(task); scene_idx.append(s)
    return (torch.tensor(np.stack(images)).permute(0, 3, 1, 2).contiguous(),
            torch.tensor(ids), torch.tensor(masks), np.array(tasks), torch.tensor(scene_idx))

train = build_split(20_000, seed=1)
test = build_split(2_000, seed=2)
print("train:", len(train[0]), "scenes,", len(train[1]), "text examples;  test:", len(test[0]), "scenes,", len(test[1]), "text examples")
print("example:", " ".join(vocab[i] for i in train[1][1] if i != 0))
```

```text
vocabulary: 35 words
train: 20000 scenes, 120000 text examples;  test: 2000 scenes, 12000 text examples
example: how many shapes ? two <eos>
```

The vocabulary is just 35 words. The data set has 20,000 training scenes with six text examples each, and 2,000 held-out test scenes. A few training images and their captions:

```python
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for ax, i in zip(axes.ravel(), range(8)):
    ax.imshow(train[0][i].permute(1, 2, 0)); ax.axis("off")
    cap = " ".join(vocab[j] for j in train[1][6 * i].tolist() if j > 1 and j != stoi["<cap>"])
    ax.set_title(cap.replace(" ,", ",").replace(" .", "."), fontsize=8)
fig.tight_layout(); fig.savefig("figures/vlm_data.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/vlm/vlm_data.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Eight training images with their captions. Shapes sit in a 3 by 3 grid of cells, one shape of each kind at most, in one of four colors." %}

## The model

The model has three parts, and each is small.

**The vision encoder** is a miniature vision transformer [[5]](#references). It cuts the image into $$8 \times 8$$ pixel patches, giving a grid of $$6 \times 6 = 36$$ patches. A single convolution with stride 8 turns each patch into a vector, which is the same as applying one shared linear map to every flattened patch. A learned position embedding is added so that the model knows where each patch came from, and two transformer blocks let the patches exchange information. The attention here is *not* causal: every patch can look at every other. The output is 36 vectors of width 64.

**The projector** is a two-layer network that maps each 64-dimensional visual vector into the 128-dimensional embedding space of the language model. In LLaVA the first version of this was a single linear layer, and a two-layer MLP was found to work better [[6]](#references). The projector is where the visual world is translated into the language model's input space, and it is typically the only part that is trained from scratch when the encoder and the language model are both pretrained.

**The language model** is a small decoder-only transformer with causal attention, like the one in the transformer post. Its input sequence is the 36 projected visual tokens followed by the text tokens of the question and, during training, the answer. It uses one positional table for the whole sequence, so that visual tokens occupy positions 0 to 35, and text begins at position 36.

During training we compute the usual next-token cross-entropy but only on the **answer** tokens, not on the question. The model is not being trained to predict questions. The code makes this explicit with a loss mask.

```python
class Block(nn.Module):
    def __init__(self, d, heads, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)    # (B, H, T, dh)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))

class VisionEncoder(nn.Module):
    """A small ViT: cut the image into 8x8 patches, embed them, let them attend to each other."""
    def __init__(self, d=64, layers=2, patch=8):
        super().__init__()
        self.embed = nn.Conv2d(3, d, patch, stride=patch)                   # one linear map per patch
        self.pos = nn.Parameter(torch.zeros(1, (SIZE // patch) ** 2, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, 4, causal=False) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
    def forward(self, img):
        x = self.embed(img.float() / 127.5 - 1).flatten(2).transpose(1, 2) + self.pos   # (B, 36, d)
        for b in self.blocks:
            x = b(x)
        return self.ln(x)

class TinyVLM(nn.Module):
    def __init__(self, vocab_size, d_vis=64, d=128, layers=4, n_patches=36):
        super().__init__()
        self.n_patches = n_patches
        self.vision = VisionEncoder(d_vis)
        self.projector = nn.Sequential(nn.Linear(d_vis, d), nn.GELU(), nn.Linear(d, d))   # vision space -> language space
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(n_patches + TEXT_LEN, d)
        self.blocks = nn.ModuleList([Block(d, 4, causal=True) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        for emb in (self.tok, self.pos):                                     # small initial embeddings -> near-uniform first predictions
            nn.init.normal_(emb.weight, std=0.02)

    def visual_tokens(self, img):
        return self.projector(self.vision(img))                              # (B, 36, d)

    def forward(self, text, vis=None, keep=None):
        """text: (B, T) word ids. vis: (B, k, d) visual tokens, with their original patch indices in keep (B, k)."""
        B, T = text.shape
        if keep is None:
            keep = torch.arange(self.n_patches, device=text.device).expand(B, -1)
        k = vis.size(1)
        text_pos = self.n_patches + torch.arange(T, device=text.device)
        x = torch.cat([vis + self.pos(keep), self.tok(text) + self.pos(text_pos)], dim=1)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x[:, k:]))                                  # logits at the text positions

torch.manual_seed(0)
model = TinyVLM(len(vocab)).to(device)
n_vis = sum(p.numel() for p in model.vision.parameters()) + sum(p.numel() for p in model.projector.parameters())
print(f"parameters: {sum(p.numel() for p in model.parameters()):,} "
      f"(vision encoder + projector: {n_vis:,}; language model: {sum(p.numel() for p in model.parameters()) - n_vis:,})")
```

```text
parameters: 945,088 (vision encoder + projector: 139,584; language model: 805,504)
```

The whole model has 945,088 parameters. Of those, 139,584 belong to the vision encoder and projector, and 805,504 to the language model.

One detail worth pointing out is that each visual token carries its *original patch index* through to the positional lookup. This is not needed for the baseline, where all 36 tokens are present in order. It becomes important in the last experiment, where we remove some tokens and need the remaining ones to keep their identity.

## Training

We train the vision encoder, the projector, and the language model together, from random initialization, for 6,000 steps with AdamW and a one-cycle learning rate schedule. Every step takes a batch of 128 text examples, looks up the image for each, and computes the loss on the answer tokens.

```python
imgs_tr, ids_tr, mask_tr, tasks_tr, sidx_tr = [t.to(device) if torch.is_tensor(t) else t for t in train]
steps, batch_size, lr = 6000, 128, 1e-3
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
history, start = [], time.time()
torch.manual_seed(0)
for step in range(1, steps + 1):
    idx = torch.randint(len(ids_tr), (batch_size,), device=device)
    text, mask = ids_tr[idx], mask_tr[idx]
    vis = model.visual_tokens(imgs_tr[sidx_tr[idx]])
    logits = model(text[:, :-1], vis)                                        # predict word j+1 from words up to j
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1), reduction="none")
    loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()        # average over answer tokens only
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    history.append(loss.item())
    if step in (1, 250, 500, 1000, 2000, 4000, 6000):
        print(f"step {step:5d} | loss {np.mean(history[-50:]):.4f} | {time.time() - start:4.0f}s")
```

```text
step     1 | loss 3.6353 |    2s
step   250 | loss 0.4845 |   52s
step   500 | loss 0.3046 |  111s
step  1000 | loss 0.1183 |  227s
step  2000 | loss 0.0274 |  462s
step  4000 | loss 0.0005 |  846s
step  6000 | loss 0.0000 | 1134s
```

The loss starts near $$\ln 35 \approx 3.6$$, which is what a uniform guess over the vocabulary would give, falls quickly during the first thousand steps, and ends at essentially zero.

```python
fig, ax = plt.subplots(figsize=(6.5, 3.4))
sm = np.convolve(history, np.ones(25) / 25, mode="valid")
ax.plot(history, color="#c6dbef", lw=0.6); ax.plot(np.arange(len(sm)) + 12, sm, color="#08519c", lw=1.8)
ax.set_xlabel("training step"); ax.set_ylabel("answer-token loss"); ax.set_yscale("log")
fig.tight_layout(); fig.savefig("figures/vlm_loss.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/vlm/vlm_loss.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Training loss on answer tokens, on a log scale. The pale line is the raw loss at each step; the dark line is a 25-step average." %}

## Does it work?

We evaluate on the 2,000 held-out scenes with greedy decoding: the model is given an image and a question, and writes the most likely next word repeatedly until it emits an end-of-sequence token. An answer counts as correct only if the whole generated string matches the target exactly. A caption is correct only if every word is right.

```python
imgs_te, ids_te, mask_te, tasks_te, sidx_te = [t.to(device) if torch.is_tensor(t) else t for t in test]

@torch.no_grad()
def answer(prompt_ids, vis, keep=None, max_new=14):
    """Greedy decoding: append the most likely word until every sequence has produced <eos>."""
    text = prompt_ids
    done = torch.zeros(len(text), dtype=torch.bool, device=device)
    out = []
    for _ in range(max_new):
        nxt = model(text, vis, keep)[:, -1].argmax(-1)
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        out.append(nxt); text = torch.cat([text, nxt[:, None]], dim=1)
        done |= nxt == stoi["<eos>"]
        if done.all():
            break
    return torch.stack(out, dim=1)

def accuracy_by_task(vis_fn=None, bs=500):
    """Exact-match accuracy per task on the test set. vis_fn(images) -> (visual tokens, patch indices)."""
    model.eval()
    results = {}
    for task in ["caption", "count", "color", "where", "exist"]:
        rows = torch.tensor(np.where(tasks_te == task)[0], device=device)
        correct = 0
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            text, mask = ids_te[r], mask_te[r]
            n_prompt = int(mask[0].nonzero()[0])                  # the answer starts where the loss mask turns on
            vis, keep = vis_fn(imgs_te[sidx_te[r]]) if vis_fn else (model.visual_tokens(imgs_te[sidx_te[r]]), None)
            pred = answer(text[:, :n_prompt], vis, keep)
            gold = text[:, n_prompt:n_prompt + pred.size(1)] * mask[:, n_prompt:n_prompt + pred.size(1)]
            correct += (pred == gold).all(dim=1).sum().item()
        results[task] = correct / len(rows)
    model.train()
    return results

with torch.no_grad():
    base = accuracy_by_task()
print("task      exact-match accuracy on held-out scenes")
for k, v in base.items():
    print(f"{k:8s}  {100 * v:5.1f}%")
```

```text
task      exact-match accuracy on held-out scenes
caption   100.0%
count     100.0%
color     100.0%
where     100.0%
exist     100.0%
```

Perfect scores on every task. That deserves some suspicion before celebration, so let us look at what the number does and does not mean. First, this is a very easy world: the shapes are flat colors on a white background, the grid is regular, and the model is trained on 120,000 examples of a language with 35 words. A model of this size is expected to learn it. The point of the test is to show that the pipeline works, not that the model is capable. Second, the held-out scenes are new images, but many of the underlying layouts and color combinations appear in the training set in other jitter positions, so the test measures generalization to new pixel arrangements and not to new concepts. A harder test, such as withholding certain color and shape combinations during training, would be more demanding, and is a good exercise at the end.

What we can check is whether the model actually uses the image. The control is to give each question the image of a *different* scene. If the model were answering from the text alone, using the statistics of questions and answers, accuracy would not change. If it reads the image, accuracy should fall to the level of guessing.

```python
# Control: show the question with the image of a *different* scene. If the model really reads the image,
# its accuracy should fall to the level of an educated guess.
def shuffled_vis(images):
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(len(images), generator=g).to(device)
    return model.visual_tokens(images[perm]), None

with torch.no_grad():
    blind = accuracy_by_task(shuffled_vis)
chance = {"caption": None, "count": 1 / 3, "color": 1 / 4, "where": 1 / 9, "exist": 1 / 2}
print("task      with the image   shown a different image   guess-level accuracy")
for k in base:
    ch = f"{100 * chance[k]:.0f}%" if chance[k] else "—"
    print(f"{k:8s}  {100 * base[k]:8.1f}%      {100 * blind[k]:8.1f}%                 {ch}")
```

```text
task      with the image   shown a different image   guess-level accuracy
caption      100.0%           1.4%                 —
count        100.0%          33.0%                 33%
color        100.0%          25.3%                 25%
where        100.0%          10.5%                 11%
exist        100.0%          56.1%                 50%
```

```python
fig, ax = plt.subplots(figsize=(6.5, 3.4))
tasks_ = ["caption", "count", "color", "where", "exist"]
x = np.arange(len(tasks_))
ax.bar(x - 0.2, [100 * base[t] for t in tasks_], 0.4, color="#08519c", label="with the image")
ax.bar(x + 0.2, [100 * blind[t] for t in tasks_], 0.4, color="#d95f02", label="shown a different image")
ax.set_xticks(x); ax.set_xticklabels(tasks_); ax.set_ylabel("exact-match accuracy (%)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/vlm_blind.png", dpi=170, bbox_inches="tight")
```

The captions go from 100% to 1.4%, and the other tasks fall to roughly what random guessing gives: a third for counting three possible numbers, a quarter for four colors, a ninth for nine locations. The existence question falls to 56% instead of 50%, since the answer "yes" is a little more common than "no" in the data. The model answers from the image, and not from the question alone.

{% include figure.liquid path="assets/img/blog/vlm/vlm_blind.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Accuracy with the correct image (blue) and with the image of a different scene (orange). The collapse in the orange bars shows that the answers depend on the image." %}

Here are six test images with the model's answers to four of their questions.

```python
model.eval()
rng = np.random.default_rng(7)
picks = rng.choice(len(imgs_te), 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for ax, s in zip(axes.ravel(), picks):
    img = imgs_te[s:s + 1]
    rows = np.where(sidx_te.cpu().numpy() == s)[0]
    lines = []
    with torch.no_grad():
        vis = model.visual_tokens(img)
        for r in rows[:4]:
            text, mask = ids_te[r:r + 1], mask_te[r:r + 1]
            n_prompt = int(mask[0].nonzero()[0])
            pred = answer(text[:, :n_prompt], vis)[0].tolist()
            q = " ".join(vocab[i] for i in text[0, :n_prompt].tolist())
            a = " ".join(vocab[i] for i in pred if i > 1)
            lines.append(f"{q if q != '<cap>' else 'caption:'}  →  {a}")
    ax.imshow(img[0].permute(1, 2, 0).cpu()); ax.axis("off")
    ax.set_title("\n".join(l.replace(" ,", ",").replace(" .", ".").replace(" ?", "?") for l in lines), fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("figures/vlm_samples.png", dpi=170, bbox_inches="tight")
model.train()
print("saved")
```

{% include figure.liquid path="assets/img/blog/vlm/vlm_samples.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Model outputs on six held-out scenes. Each panel shows the caption the model wrote and its answers to three questions. All are correct." %}

## The cost of visual tokens

Now the part of this toy that carries over to real systems. In our model, the image becomes 36 tokens, so the language model's input is the 36 visual tokens plus about ten text tokens. In a real VLM the number is much larger. LLaVA-1.5 uses a 336-pixel image cut into $$14 \times 14$$ patches, which gives $$24 \times 24 = 576$$ visual tokens for a single image [[6]](#references). Models that handle high-resolution images or several images, or video, can use thousands. Those tokens sit in the language model's context for the whole answer. They add to the attention computation, they occupy space in the key-value cache, and the cache is already the main memory cost of generation, as discussed in the [post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}). In many VLMs, image tokens far outnumber the text tokens, so reducing them is attractive.

This motivates **visual token pruning**: keep only some of the visual tokens before, or inside, the language model. Published approaches include dropping tokens by attention score inside the language model, as in FastV [[7]](#references), and merging similar tokens inside the vision encoder, as in ToMe [[8]](#references). Our toy lets us ask the basic question directly. How many of the 36 tokens does the model need, and which ones should we keep?

We try two rules, each keeping $$k$$ of the 36 tokens. **Random** keeps a random subset. **Content-based** keeps the $$k$$ patches with the most non-white pixels, a stand-in for a saliency score. In our images only a fraction of the patches contain any part of a shape: on the test set, an average of 7.9 of the 36, and at most 12. If the model only needs the patches with shapes in them, then the content rule should do very well down to about 12 tokens. The tokens keep their original position indices.

```python
# Visual token pruning at inference time: keep only k of the 36 visual tokens before they enter the language model.
patch_mass = lambda images: (255 - images.float()).mean(dim=1).view(-1, 6, 8, 6, 8).mean(dim=(2, 4)).flatten(1)   # ink per patch

def make_pruner(k, mode):
    def fn(images):
        vis = model.visual_tokens(images)
        if k >= 36:
            return vis, None
        if mode == "random":
            scores = torch.rand(len(images), 36, device=device, generator=None)
        else:
            scores = patch_mass(images)                      # keep the patches that contain something
        keep = scores.topk(k, dim=1).indices.sort(dim=1).values
        return torch.gather(vis, 1, keep[:, :, None].expand(-1, -1, vis.size(2))), keep
    return fn

ks = [36, 24, 16, 12, 8, 6, 4, 2]
torch.manual_seed(0)
prune_results = {"random": [], "content-based": []}
with torch.no_grad():
    for k in ks:
        for mode, name in (("random", "random"), ("saliency", "content-based")):
            res = accuracy_by_task(make_pruner(k, mode))
            prune_results[name].append(res)
print("kept tokens   mean accuracy over count/color/where/exist:  random   content-based")
for i, k in enumerate(ks):
    r = np.mean([prune_results["random"][i][t] for t in ("count", "color", "where", "exist")])
    c = np.mean([prune_results["content-based"][i][t] for t in ("count", "color", "where", "exist")])
    print(f"{k:11d}   {100 * r:44.1f}%  {100 * c:9.1f}%")
ink = patch_mass(imgs_te).gt(1.0).sum(dim=1).float()
print(f"patches that contain any shape pixels: mean {ink.mean():.1f} of 36 (min {int(ink.min())}, max {int(ink.max())})")
```

```text
kept tokens   mean accuracy over count/color/where/exist:  random   content-based
         36                                          100.0%      100.0%
         24                                           86.4%       92.6%
         16                                           73.9%       79.8%
         12                                           66.5%       68.5%
          8                                           56.2%       61.7%
          6                                           52.2%       59.8%
          4                                           44.8%       53.0%
          2                                           36.2%       48.0%
patches that contain any shape pixels: mean 7.9 of 36 (min 3, max 12)
```

Three things stand out. First, accuracy drops steeply for both rules as tokens are removed, far more than the arithmetic of "only 8 patches contain a shape" suggests. With 12 tokens kept, which is enough to cover every shape in every test image, the content rule gets 68.5% of questions right and random gets 66.5%. The model is evidently using the *empty* patches too. This is plausible: after two transformer layers of attention, each patch vector has mixed in information from the rest of the image, and a trained network can use some tokens as a place to keep global information, as has been observed in larger vision transformers [[9]](#references). Removing them removes information that the language model has learned to expect.

Second, content-based selection beats random for every budget here, which is what intuition suggests.

Third, and most important, the language model was never trained to see fewer tokens. Every training example had exactly 36. So these numbers describe how a model fails under a change it was not prepared for, not how well one could do. We can test that by fine-tuning the same model for 1,500 more steps, with a random number of tokens kept at each step, between 2 and 36.

```python
# The model never saw a missing token during training. Fine-tune the same model for a short while with random
# visual-token dropout (a random number of tokens kept, between 2 and 36) so that the language model learns to cope.
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
before_ft = {name: [dict(r) for r in rs] for name, rs in prune_results.items()}
torch.manual_seed(1)
ft_steps = 1500
for step in range(1, ft_steps + 1):
    idx = torch.randint(len(ids_tr), (batch_size,), device=device)
    text, mask = ids_tr[idx], mask_tr[idx]
    vis_all = model.visual_tokens(imgs_tr[sidx_tr[idx]])
    k = int(torch.randint(2, 37, (1,)))
    keep = torch.rand(batch_size, 36, device=device).topk(k, dim=1).indices.sort(dim=1).values
    vis = torch.gather(vis_all, 1, keep[:, :, None].expand(-1, -1, vis_all.size(2)))
    logits = model(text[:, :-1], vis, keep)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1), reduction="none")
    loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step in (1, 500, 1000, 1500):
        print(f"fine-tuning step {step:4d} | loss {loss.item():.4f}")
```

```text
fine-tuning step    1 | loss 2.9462
fine-tuning step  500 | loss 0.0114
fine-tuning step 1000 | loss 0.0413
fine-tuning step 1500 | loss 0.1057
```

```text
kept tokens   before fine-tuning: random  content-based   after fine-tuning: random  content-based
         36                    100.0%        100.0%                     99.7%         99.7%
         24                     86.4%         92.6%                     98.3%         98.0%
         16                     73.9%         79.8%                     95.3%         93.2%
         12                     66.5%         68.5%                     91.9%         92.1%
          8                     56.2%         61.7%                     87.3%         81.9%
          6                     52.2%         59.8%                     84.2%         80.1%
          4                     44.8%         53.0%                     79.6%         66.1%
          2                     36.2%         48.0%                     73.8%         58.1%
```

```python
fig, ax = plt.subplots(figsize=(7.5, 4))
for name, color in (("random", "#d95f02"), ("content-based", "#08519c")):
    before = [100 * mean4(r) for r in before_ft[name]]
    after = [100 * mean4(r) for r in prune_results[name]]
    ax.plot(ks, before, "o--", color=color, lw=1.5, alpha=0.7, label=f"{name}, original model")
    ax.plot(ks, after, "o-", color=color, lw=2, label=f"{name}, after token-dropout fine-tuning")
ax.invert_xaxis(); ax.set_xlabel("visual tokens kept (of 36)"); ax.set_ylabel("mean accuracy (%)")
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig("figures/vlm_pruning.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/vlm/vlm_pruning.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Mean accuracy (count, color, where, existence) as the number of visual tokens kept falls from 36 to 2. Dashed lines: the original model. Solid lines: after 1,500 steps of fine-tuning with random token dropout. Orange selects tokens at random, blue by pixel content." %}

The change is large. With 12 of 36 tokens, accuracy goes from about 67% to about 92%, and with 24 tokens it is 98%. Once the model has been given a chance to adapt, keeping 24 of the 36 tokens costs under two points of accuracy, and keeping only 12 costs about eight. At the extreme, with only 2 tokens, random selection still reaches 74% of questions, since a pair of tokens that have attended to the whole image carry a good deal of information, and two random patches often capture something.

The fine-tuned model also contains a result that is easy to miss and informative. After fine-tuning, **random selection beats content-based selection** at small budgets: 79.6% against 66.1% with 4 tokens, and 73.8% against 58.1% with 2. Before fine-tuning, the content rule had been better. The likely reason has two parts. The model was fine-tuned on random subsets, so a content-selected subset is unfamiliar to it, which is a mismatch between how the model was prepared and how tokens are chosen at test time. And ranking patches by how much ink they contain is not a good way to cover a scene: squares have far more ink per patch than circles or triangles, so a small budget can be spent entirely on one or two shapes, leaving the model blind to the others. I did not run an experiment to separate these two effects, so treat this as a likely explanation and not as a result. The practical lesson is the same either way. A pruning rule should match how the model was trained, and a rule based only on importance can lose to a random one when it ignores coverage.

## The contrastive alternative

Before the summary, here is the other family in code. CLIP trains an image encoder and a text encoder so that, among a batch of $$N$$ image-text pairs, each image is most similar to its own caption and each caption to its own image. If $$u_i$$ and $$v_j$$ are the normalized image and text embeddings, the loss for the image side is a cross-entropy over the batch:

$$
\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(u_i \cdot v_i / \tau)}{\sum_{j=1}^{N} \exp(u_i \cdot v_j / \tau)},
$$

where $$\tau$$ is a temperature. The text side is the same with the roles swapped, and the two are averaged [[1]](#references). Each example is a classification problem in which the other $$N - 1$$ items in the batch serve as negatives. In code, it is short:

```python
import math, torch
import torch.nn.functional as F

def clip_loss(img_emb, txt_emb, temperature=0.07):
    """Symmetric contrastive loss: matching image-text pairs are the 'correct class' in a batch of N pairs."""
    img = F.normalize(img_emb, dim=-1)
    txt = F.normalize(txt_emb, dim=-1)
    logits = img @ txt.T / temperature              # (N, N): entry (i, j) is the similarity of image i and text j
    labels = torch.arange(len(img))                 # the right partner of image i is text i
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

torch.manual_seed(0)
N, d = 64, 128
a, b = torch.randn(N, d), torch.randn(N, d)
print(f"unrelated embeddings, temperature 1: loss {clip_loss(a, b, 1.0):.3f}  (chance level ln N = {math.log(N):.3f})")
print(f"identical embeddings, temperature 0.07: loss {clip_loss(a, a):.4f}")
print(f"identical embeddings, temperature 1: loss {clip_loss(a, a, 1.0):.3f}")
```

```text
unrelated embeddings, temperature 1: loss 4.162  (chance level ln N = 4.159)
identical embeddings, temperature 0.07: loss 0.0001
identical embeddings, temperature 1: loss 3.190
```

The checks show what to expect. With unrelated embeddings the model is guessing among 64 partners and the loss is $$\ln 64 \approx 4.16$$. With perfectly matched embeddings and a low temperature the loss is essentially zero. And the temperature matters: with $$\tau = 1$$ even perfect matches give a high loss, because the softmax over cosine similarities, which lie in $$[-1, 1]$$, cannot become sharp. CLIP learns $$\tau$$ as a parameter instead of fixing it [[1]](#references). A later variant, SigLIP, replaces the softmax over the batch with an independent sigmoid loss on each pair [[10]](#references).

## What real VLMs add

Our model has the full structure of a prefix-style VLM, and the real ones differ mostly in scale and in how they are trained.

- **Pretrained parts.** The image encoder is typically a large vision transformer pretrained with a contrastive objective, and the language model is a pretrained LLM. Only the connection between them, and sometimes a fine-tuning stage over the whole system, is trained on the vision-language data.
- **Two-stage training.** A common recipe first trains only the projector on image-caption pairs, so that visual features land in a region of the embedding space the language model can use, and then trains the projector and the language model on instruction-following data about images [[2]](#references) [[6]](#references).
- **Resolution and layout.** Real images are larger, come in different shapes, and often contain text. Many systems cut them into tiles, each processed as its own set of tokens, which makes the token budget discussed above a first-order concern.
- **Known weaknesses.** VLMs still describe objects that are not there, miscount, and struggle with fine spatial relationships. Our toy world is built to avoid exactly those difficulties, which is why it is not a measure of real capability.

## Common implementation errors

- **Computing the loss on the question as well as the answer.** The model then spends capacity learning to predict prompts, and the reported loss is hard to interpret. Mask the prompt.
- **Misaligned shifts.** The logits at position $$t$$ predict the token at $$t+1$$. With a visual prefix of length $$k$$, the slice of the output that corresponds to text starts at index $$k$$, and an off-by-one here still trains but quietly learns the wrong thing.
- **Changing image normalization between training and evaluation.** Pixel scaling is part of the model. If the evaluation images are scaled differently, accuracy drops for reasons that have nothing to do with learning.
- **Losing positions after pruning.** If tokens are dropped and the survivors are renumbered, the position information changes. We keep each token's original index.
- **Evaluating only with the image present.** Always run the control with a mismatched or blank image. A model that has learned to answer from text priors can look good on a benchmark where questions leak the answers.
- **Testing on the training scenes.** Keep a held-out set of images, and if you can, hold out combinations of attributes as well.

## Exercises

1. **Compositional generalization.** Remove all training images that contain a yellow circle, then ask the trained model about yellow circles in the test set. Does the model compose its notion of "yellow" and "circle"?
2. **Frozen parts.** Freeze the language model and train only the projector and encoder, or the reverse. How much of the work does each part do?
3. **Fewer patches.** Use $$16 \times 16$$ patches, giving 9 tokens, and find where accuracy begins to fall.
4. **Contrastive pretraining.** Train the vision encoder first with the contrastive loss on the captions, then use it in the generative model. Does it train faster?

## References

1. A. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." *ICML*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. H. Liu, C. Li, Q. Wu, and Y. J. Lee. "Visual Instruction Tuning." *NeurIPS*, 2023. [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
3. J.-B. Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning." *NeurIPS*, 2022. [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
4. J. Li, D. Li, S. Savarese, and S. Hoi. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." *ICML*, 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
5. A. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
6. H. Liu, C. Li, Y. Li, and Y. J. Lee. "Improved Baselines with Visual Instruction Tuning." *CVPR*, 2024. [arXiv:2310.03744](https://arxiv.org/abs/2310.03744)
7. L. Chen et al. "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models." *ECCV*, 2024. [arXiv:2403.06764](https://arxiv.org/abs/2403.06764)
8. D. Bolya, C.-Y. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman. "Token Merging: Your ViT But Faster." *ICLR*, 2023. [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)
9. T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. "Vision Transformers Need Registers." *ICLR*, 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
10. X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. "Sigmoid Loss for Language Image Pre-Training." *ICCV*, 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
