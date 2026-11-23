---
layout: post
title: "Generative Adversarial Networks from Scratch: A Two-Player Game for Learning Distributions"
date: 2025-01-26 09:00:00-0600
description: Build a GAN in PyTorch, derive what the adversarial game optimizes, watch it learn a toy distribution and handwritten digits, and look honestly at where it fails.
tags: gan generative-models pytorch tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

Most of machine learning is about prediction: given an image, say what is in it. Generative modeling asks for the opposite. Given a collection of examples, produce new ones that look like they came from the same source. That sounds vague until you try to say what "look like" means. For handwritten digits there is no formula for the probability of a particular grid of pixels, and writing one down by hand is hopeless.

Generative adversarial networks, introduced by Goodfellow and colleagues in 2014 [[1]](#references), sidestep the problem. They never write down a probability. Instead, two networks play a game. One of them, the **generator**, turns random noise into candidate examples. The other, the **discriminator**, looks at examples and tries to say which ones are real and which ones the generator made. The generator improves by fooling the discriminator, and the discriminator improves by not being fooled.

In this post we build that game from the ground up. We work out what it optimizes, train a small GAN on a two-dimensional toy problem where we can see everything, and then train a convolutional GAN on handwritten digits. Along the way we will see what makes GANs hard to train, with experiments that show the problems and not just a list of warnings. The code is plain PyTorch. If you are new to neural networks, [the earlier post on building one from scratch]({% post_url 2024-11-10-neural-network-from-scratch %}) covers the basics we assume here.

## The setup

Let $$p_{\text{data}}$$ be the distribution that the training examples come from. We cannot write it down, but we can draw samples from it: the dataset. We want a model from which we can also draw samples, and which produces the same kinds of things.

The **generator** $$G_\theta$$ is a neural network that takes a random vector $$z$$, drawn from a simple distribution such as a standard Gaussian, and maps it to a data point $$G_\theta(z)$$. Because $$z$$ is random, the outputs are random too, and together they follow some distribution $$p_g$$ that depends on $$\theta$$. Learning means adjusting $$\theta$$ until $$p_g$$ is close to $$p_{\text{data}}$$. Nothing in this definition requires us to ever compute a density: we only sample.

The **discriminator** $$D_\phi$$ is an ordinary classifier. It takes a data point $$x$$ and outputs a number $$D_\phi(x) \in (0, 1)$$, interpreted as the probability that $$x$$ is a real example and not one made by $$G$$.

The two are trained against each other with the objective

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log\big(1 - D(G(z))\big)\big].
$$

Read it from the discriminator's side first. For fixed $$G$$, the discriminator wants $$V$$ large, which means $$D(x)$$ close to 1 on real data and $$D(G(z))$$ close to 0 on fakes. That is just the log-likelihood of a binary classifier. The generator wants the opposite: it chooses $$G$$ to make $$V$$ small, by making $$D(G(z))$$ large.

## What the game optimizes

It is worth knowing what this objective actually does, because it explains both why the idea works and why training is delicate. We can answer the question exactly for an idealized discriminator.

Fix the generator, so that $$p_g$$ is fixed. Write the expectation over $$z$$ as an expectation over generated samples $$x \sim p_g$$. Then

$$
V(D, G) = \int \Big( p_{\text{data}}(x) \log D(x) + p_g(x) \log\big(1 - D(x)\big) \Big)\, dx.
$$

The integrand can be maximized separately at each point $$x$$. For numbers $$a, b > 0$$, the function $$f(y) = a \log y + b \log(1 - y)$$ has derivative $$a/y - b/(1-y)$$, which is zero at $$y = a/(a+b)$$, and this is a maximum. So the best possible discriminator is

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}.
$$

This has a clear meaning. Where real data is much more likely than fakes, $$D^*$$ is near 1. Where the generator produces things that real data never does, it is near 0. Where the two densities agree, it is exactly one half.

Now plug $$D^*$$ back in to see what the generator is minimizing:

$$
\begin{aligned}
C(G) = \max_D V(D, G) &= \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right] \\
&= -\log 4 + \mathrm{KL}\!\left(p_{\text{data}} \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\right) + \mathrm{KL}\!\left(p_g \,\Big\|\, \tfrac{p_{\text{data}} + p_g}{2}\right) \\
&= -\log 4 + 2\, \mathrm{JSD}\big(p_{\text{data}} \,\|\, p_g\big).
\end{aligned}
$$

The last line uses the Jensen–Shannon divergence, a symmetric measure of the distance between two distributions. It is never negative and equals zero only when the two distributions are identical. So the generator's objective, when the discriminator is perfect, is minimized exactly when $$p_g = p_{\text{data}}$$, and at that point $$D^*(x) = 1/2$$ everywhere: the discriminator can do no better than guessing [[1]](#references).

That is the whole theoretical story, and it is a good one. It is also an idealization, and it hides three assumptions. The discriminator is assumed to be optimal at every step, while in practice it is a finite network trained for a handful of gradient steps between each generator update. The networks are assumed to be able to represent any function, while in practice they are small. And the analysis says nothing about whether gradient descent on the two players *converges* to the equilibrium it describes. Simultaneous gradient descent on a two-player game can circle around an equilibrium forever without reaching it [[3]](#references). Most of the practical difficulty of GANs lives in the gap between this clean picture and what actually happens.

### The loss that is used in practice

There is one modification that nearly everyone makes. Early in training, the generator is poor and the discriminator easily recognizes its output, so $$D(G(z))$$ is close to 0. The generator's loss $$\log(1 - D(G(z)))$$ is then nearly flat in that region, and its gradient is tiny: the generator gets almost no signal exactly when it needs the most. The standard fix, proposed in the original paper, is to have the generator maximize $$\log D(G(z))$$ instead. The fixed point is the same, but the gradients are large when the generator is losing. This is the **non-saturating loss**, and in code it is simply a binary cross-entropy against the label "real" for the generated samples.

## Training: two optimizers, alternating

Each training step does two updates. First, the discriminator is trained on a batch of real data (target 1) and a batch of generated data (target 0). Then the generator is trained to make the discriminator say 1 on its new samples. In PyTorch the one detail that matters is `detach()`: when updating the discriminator we must not backpropagate into the generator, so the fake batch is detached from the generator's computation graph. The generator update, on the other hand, must flow through the discriminator to reach the generator's weights, though only the generator's optimizer takes a step.

## Experiment 1: eight Gaussians on a ring

Before images, a problem we can fully visualize. The data are 2-D points drawn from a mixture of eight Gaussians, equally spaced on a circle of radius 2, each with standard deviation 0.1. It is a small distribution with a clearly multimodal shape, and a model that learns it must produce all eight clusters with roughly equal weight.

We will measure two things about the generated samples: how many of the eight modes are **covered** (a mode counts if at least 1% of the samples land within three standard deviations of its center), and what fraction of all samples land near any mode at all, a rough measure of sample quality.

```python
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)

def ring_centers(modes=8, radius=2.0):
    angles = torch.arange(modes) * 2 * math.pi / modes
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)

def grid_centers(k=5, spacing=2.0):
    xs = (torch.arange(k) - (k - 1) / 2) * spacing
    return torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)

def sample_mixture(n, centers, std):
    """n points from an equal-weight mixture of Gaussians with the given centers."""
    return centers[torch.randint(len(centers), (n,))] + std * torch.randn(n, 2)

def mode_metrics(samples, centers, std):
    """(modes covered, fraction of samples within 3 std of a mode).
    A mode counts as covered if at least 1% of the samples land close to it."""
    d = torch.cdist(samples, centers)
    close = d.min(dim=1).values < 3 * std
    counts = torch.bincount(d.argmin(dim=1)[close], minlength=len(centers))
    return int((counts >= 0.01 * len(samples)).sum()), close.float().mean().item()
```

The generator and discriminator are both small multilayer perceptrons. The generator maps a two-dimensional Gaussian vector to a 2-D point. The training function takes snapshots of the generator along the way so that we can watch it learn.

```python
def make_mlps(width=128, zdim=2):
    G = nn.Sequential(nn.Linear(zdim, width), nn.ReLU(),
                      nn.Linear(width, width), nn.ReLU(),
                      nn.Linear(width, 2))
    D = nn.Sequential(nn.Linear(2, width), nn.LeakyReLU(0.2),
                      nn.Linear(width, width), nn.LeakyReLU(0.2),
                      nn.Linear(width, 1))
    return G, D

def train_toy(centers, std, seed, steps=6000, batch=256, lr=1e-3, snapshots=()):
    torch.manual_seed(seed)
    G, D = make_mlps()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()                      # takes logits; applies the sigmoid itself
    ones, zeros = torch.ones(batch, 1), torch.zeros(batch, 1)
    saved = {}
    for step in range(steps + 1):
        if step in snapshots:
            with torch.no_grad():
                saved[step] = G(torch.randn(2000, 2))
        real = sample_mixture(batch, centers, std)
        fake = G(torch.randn(batch, 2))
        # Discriminator step: real -> 1, fake -> 0. detach() keeps this step from touching G.
        loss_d = bce(D(real), ones) + bce(D(fake.detach()), zeros)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        # Generator step (non-saturating loss): make D label the fakes as real.
        loss_g = bce(D(fake), ones)
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    with torch.no_grad():
        final = G(torch.randn(5000, 2))
    return G, D, final, saved

centers8, std8 = ring_centers(), 0.1
snap_steps = (0, 100, 500, 2000, 6000)
G8, D8, final8, saved8 = train_toy(centers8, std8, seed=0, snapshots=snap_steps)
for step in snap_steps:
    covered, close = mode_metrics(saved8[step], centers8, std8)
    print(f"step {step:5d}: {covered} of 8 modes covered, {100 * close:5.1f}% of samples within 3 std of a mode")
```

```text
step     0: 0 of 8 modes covered,   0.0% of samples within 3 std of a mode
step   100: 1 of 8 modes covered,   5.7% of samples within 3 std of a mode
step   500: 8 of 8 modes covered,  23.7% of samples within 3 std of a mode
step  2000: 8 of 8 modes covered,  86.7% of samples within 3 std of a mode
step  6000: 8 of 8 modes covered,  95.0% of samples within 3 std of a mode
```

{% include figure.liquid path="assets/img/blog/gan/gan_ring.png" class="img-fluid rounded z-depth-1" zoomable=true caption="A GAN learning eight Gaussians on a ring. Gray points are real data; orange points are generator samples at five moments of training. The generator begins as a tiny blob at the center, is dragged toward the data, spreads out over the whole region, and then concentrates on the clusters." %}

```python
real = sample_mixture(2000, centers8, std8)
fig, axes = plt.subplots(1, 5, figsize=(14, 3))
for ax, step in zip(axes, snap_steps):
    ax.scatter(real[:, 0], real[:, 1], s=3, c="#bdbdbd", label="real")
    s = saved8[step]
    ax.scatter(s[:, 0], s[:, 1], s=3, c="#d95f02", label="generated")
    ax.set_title(f"step {step}"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(frameon=False, markerscale=4, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig("figures/gan_ring.png", dpi=170, bbox_inches="tight")
```

The progression tells a story that the final number hides. After 100 steps the generator has found a corner of the plane and is pushing a cloud of points toward one of the clusters. By step 500 it has covered all eight modes, but the samples are spread over a wide region, with only about a quarter of them near a mode. At step 2000 they are mostly on the ring, and by step 6000, 95% of the samples are close to a mode and all eight are covered.

Look at the last panel closely. Between neighboring clusters there are thin streaks of orange points that belong to no mode. This is not a bug in the code, and it is typical of GANs on multimodal data. The generator is a continuous function of its input. To produce several separate clusters from one connected cloud of noise, the map must stretch and tear, and the points in the stretched regions land between the clusters. The generator pays for the topological mismatch with a few stray samples.

### What the discriminator sees

The discriminator holds the other half of the picture. We can evaluate it on a grid covering the plane, to see what it believes about every location.

```python
# What has the discriminator learned? Its output D(x) = sigmoid(logit) on a grid of points.
xs = torch.linspace(-3.5, 3.5, 200)
grid = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), dim=-1).reshape(-1, 2)
with torch.no_grad():
    prob_real = torch.sigmoid(D8(grid)).reshape(200, 200).numpy()
fig, ax = plt.subplots(figsize=(4.8, 4.2))
im = ax.imshow(prob_real, extent=(-3.5, 3.5, -3.5, 3.5), origin="lower", cmap="RdBu", vmin=0, vmax=1)
ax.scatter(real[:300, 0], real[:300, 1], s=3, c="k", alpha=0.5)
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, label="D(x), the probability of 'real'")
fig.tight_layout(); fig.savefig("figures/gan_discriminator.png", dpi=170, bbox_inches="tight")
print("D at the data modes, mean:", round(float(torch.sigmoid(D8(centers8)).mean()), 3))
print("D at the ring's center (0, 0):", round(float(torch.sigmoid(D8(torch.zeros(1, 2)))), 3))
```

```text
D at the data modes, mean: 0.524
D at the ring's center (0, 0): 0.01
```

{% include figure.liquid path="assets/img/blog/gan/gan_discriminator.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The trained discriminator's output D(x) over the plane. Blue means 'looks real', red means 'looks fake'. The black dots are real data points. The discriminator is confident that the empty center and the outside are fake, and unsure near the clusters." %}

Near the data modes the discriminator outputs 0.52 on average, very close to the value of one half that the theory predicts at equilibrium: wherever the generator is doing well, the discriminator cannot tell the difference. At the center of the ring, where there is no data and the generator rarely samples, the discriminator says 0.01. Its job is to mark everything off the data as fake, and it does so well in regions the generator has left alone.

## Experiment 2: a harder target

Eight well-separated clusters turn out to be easy. A grid of 25 clusters, with a much smaller standard deviation of 0.05, is harder. We use exactly the same networks and optimizer, train for 8,000 steps, and repeat with five different random seeds, since GAN results can vary a lot from run to run.

```python
centers25, std25 = grid_centers(), 0.05
print("seed  modes covered (of 25)   samples near a mode")
results = {}
for seed in range(5):
    _, _, final, _ = train_toy(centers25, std25, seed=seed, steps=8000)
    covered, close = mode_metrics(final, centers25, std25)
    results[seed] = (covered, close, final)
    print(f"{seed:4d}  {covered:21d}   {100 * close:5.1f}%")
```

```text
seed  modes covered (of 25)   samples near a mode
   0                     15    46.2%
   1                     17    75.6%
   2                     20    76.2%
   3                     20    71.2%
   4                     11    32.1%
```

The five runs, which differ only in the random seed, cover between 11 and 20 of the 25 modes, and the fraction of samples that actually land near a mode ranges from 32% to 76%. Same code, same data, very different outcomes. Training a GAN is not a single procedure with a single result; it is a draw from a distribution of results. The figure shows the worst, the middle, and the best of the five.

```python
order = sorted(results, key=lambda s: results[s][0])
chosen = [order[0], order[len(order) // 2], order[-1]]
real25 = sample_mixture(2000, centers25, std25)
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
axes[0].scatter(real25[:, 0], real25[:, 1], s=3, c="#737373"); axes[0].set_title("real data (25 modes)")
for ax, seed in zip(axes[1:], chosen):
    covered, close, final = results[seed]
    ax.scatter(final[:, 0], final[:, 1], s=3, c="#d95f02")
    ax.set_title(f"seed {seed}: {covered} of 25 modes")
for ax in axes:
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("figures/gan_grid.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/gan/gan_grid.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Left: the real data, 25 tight clusters. The other three panels: generator samples from the worst, median, and best of five seeds. None of the generators reproduces the grid; they draw lines connecting the clusters they do find and leave several clusters empty." %}

Two failure modes appear together here. Some clusters are simply **missing**: the generator has no samples there. This is called **mode dropping**, and it is the milder cousin of **mode collapse**, in which the generator maps almost all of its noise to one or a few outputs. And many generated points lie along **lines between** clusters, the same bridging effect as in the first experiment, now much more pronounced. Neither failure is visible in the training loss, which looks perfectly ordinary in all five runs. Measuring them required knowing where the modes are, which is only possible on a toy problem.

## Experiment 3: handwritten digits with a convolutional GAN

Now images. We use MNIST, 60,000 grayscale digits of size 28 × 28, and the architecture family from the DCGAN paper [[2]](#references), which made image GANs reliably trainable. The ingredients are worth listing, since each one fixes a specific problem.

- The generator upsamples with **transposed convolutions**, going from a 7 × 7 feature map to 14 × 14 to 28 × 28, and ends with `tanh`, so its outputs lie in $$[-1, 1]$$. This is why the training images are rescaled to $$[-1, 1]$$ too. A mismatch between the output range of the generator and the range of the real data is an easy way to make the discriminator's job trivial.
- The discriminator is a small strided **convolutional network** with LeakyReLU activations, and it outputs one logit.
- **Batch normalization** in both networks stabilizes training, and weights are initialized from $$\mathcal{N}(0, 0.02^2)$$.
- The optimizer is Adam with a learning rate of $$2 \times 10^{-4}$$ and $$\beta_1 = 0.5$$. The lower momentum is an empirical finding of the DCGAN paper: a high $$\beta_1$$ makes the two-player dynamics oscillate.

```python
import torchvision
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)
train_set = torchvision.datasets.MNIST("data", train=True, download=True)
test_set = torchvision.datasets.MNIST("data", train=False, download=True)

def to_tensor(ds):
    x = ds.data.float().div(255).unsqueeze(1)          # (N, 1, 28, 28) in [0, 1]
    return x * 2 - 1, ds.targets                       # rescale to [-1, 1] to match the generator's tanh
x_train, y_train = to_tensor(train_set)
x_test, y_test = to_tensor(test_set)
x_train = x_train.to(device)
print("training images:", tuple(x_train.shape), "range", float(x_train.min()), "to", float(x_train.max()))
```

```text
device: mps
training images: (60000, 1, 28, 28) range -1.0 to 1.0
```

```python
latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * 7 * 7), nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 7x7   -> 14x14
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),     # 14x14 -> 28x28
            nn.Tanh())
    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),                      # 28x28 -> 14x14
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 14x14 -> 7x7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))                                          # one logit
    def forward(self, x):
        return self.net(x)

def init_weights(m):                                  # DCGAN convention: N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)

torch.manual_seed(0)
G, D = Generator().to(device), Discriminator().to(device)
G.apply(init_weights); D.apply(init_weights)
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"generator: {count(G):,} parameters, discriminator: {count(D):,} parameters")
z = torch.randn(4, latent_dim, device=device)
print("generator output shape:", tuple(G(z).shape), " discriminator output shape:", tuple(D(G(z)).shape))
```

```text
generator: 778,305 parameters, discriminator: 138,817 parameters
generator output shape: (4, 1, 28, 28)  discriminator output shape: (4, 1)
```

The discriminator has about 139 thousand parameters and the generator about 778 thousand. The generator is deliberately the larger network. The training loop is the alternating update we described, using binary cross-entropy on logits (which folds the sigmoid into the loss for numerical stability). We track the loss of each network and the average output of the discriminator on real and generated images, and we save samples from a fixed set of 64 noise vectors at several epochs, so that the same latent points can be followed through training.

```python
import time
epochs, batch_size = 15, 128
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
fixed_z = torch.randn(64, latent_dim, device=device)       # the same 64 noise vectors at every checkpoint
grids, log = {}, []

def save_grid(epoch):
    G.eval()
    with torch.no_grad():
        grids[epoch] = G(fixed_z).cpu()
    G.train()

save_grid(0)
start = time.time()
for epoch in range(1, epochs + 1):
    perm = torch.randperm(len(x_train), device=device)
    sums = np.zeros(4); n_batches = 0
    for i in range(0, len(perm) - batch_size + 1, batch_size):
        real = x_train[perm[i:i + batch_size]]
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = G(z)
        ones, zeros = torch.ones(batch_size, 1, device=device), torch.zeros(batch_size, 1, device=device)

        # Discriminator: push D(real) toward 1 and D(fake) toward 0.
        logits_real, logits_fake_d = D(real), D(fake.detach())
        loss_d = bce(logits_real, ones) + bce(logits_fake_d, zeros)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # Generator: push D(fake) toward 1 (the non-saturating loss).
        logits_fake = D(fake)
        loss_g = bce(logits_fake, ones)
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        sums += [loss_d.item(), loss_g.item(),
                 torch.sigmoid(logits_real).mean().item(), torch.sigmoid(logits_fake_d).mean().item()]
        n_batches += 1
    log.append(sums / n_batches)
    if epoch in (1, 2, 3, 5, 10, 15):
        save_grid(epoch)
    if epoch in (1, 2, 3, 5, 10, 15):
        print(f"epoch {epoch:2d} | D loss {log[-1][0]:.3f} | G loss {log[-1][1]:.3f} | "
              f"D(real) {log[-1][2]:.2f} | D(fake) {log[-1][3]:.2f} | {time.time() - start:4.0f}s")
```

```text
epoch  1 | D loss 0.415 | G loss 2.604 | D(real) 0.85 | D(fake) 0.15 |   40s
epoch  2 | D loss 0.495 | G loss 2.019 | D(real) 0.81 | D(fake) 0.20 |   80s
epoch  3 | D loss 0.583 | G loss 1.942 | D(real) 0.78 | D(fake) 0.22 |  120s
epoch  5 | D loss 0.725 | G loss 1.722 | D(real) 0.74 | D(fake) 0.26 |  201s
epoch 10 | D loss 0.716 | G loss 1.792 | D(real) 0.75 | D(fake) 0.25 |  475s
epoch 15 | D loss 0.664 | G loss 1.926 | D(real) 0.76 | D(fake) 0.24 |  872s
```

This ran on the GPU of a laptop (the device is printed above) and took about ten minutes for 15 epochs. Here are the generator's samples from the same 64 noise vectors at several points in training.

```python
def show_grid(ax, images, title):
    g = images.view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
    ax.imshow(g, cmap="gray", vmin=-1, vmax=1); ax.set_title(title); ax.axis("off")

shown = (0, 1, 3, 5, 15)
fig, axes = plt.subplots(1, len(shown), figsize=(14, 3.2))
for ax, e in zip(axes, shown):
    show_grid(ax, grids[e], "before training" if e == 0 else f"epoch {e}")
fig.tight_layout(); fig.savefig("figures/gan_mnist_progress.png", dpi=170, bbox_inches="tight")

log_arr = np.array(log)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
ax1.plot(range(1, epochs + 1), log_arr[:, 0], color="#08519c", lw=2, label="discriminator loss")
ax1.plot(range(1, epochs + 1), log_arr[:, 1], color="#d95f02", lw=2, label="generator loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(frameon=False)
ax2.plot(range(1, epochs + 1), log_arr[:, 2], color="#08519c", lw=2, label="D(real image)")
ax2.plot(range(1, epochs + 1), log_arr[:, 3], color="#d95f02", lw=2, label="D(generated image)")
ax2.axhline(0.5, color="gray", ls="--", lw=1)
ax2.set_xlabel("epoch"); ax2.set_ylabel("average output of D"); ax2.set_ylim(0, 1); ax2.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/gan_mnist_losses.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/gan/gan_mnist_progress.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Generator output for the same 64 noise vectors before training, and after epochs 1, 3, 5, and 15. Each noise vector keeps its place in the grid, so the same slot can be followed through training." %}

Before training the output is a flat gray image. After a single epoch there are already digit-like strokes, mostly broken and noisy. By epoch 5 most samples are recognizable as digits, and by epoch 15 the strokes are cleaner and the shapes more consistent. Many of the samples are still odd: some digits are ambiguous or malformed, which is expected from a small model trained for ten minutes.

### The losses do not tell you this

Now the same training run seen through its loss curves.

{% include figure.liquid path="assets/img/blog/gan/gan_mnist_losses.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Left: the discriminator and generator losses by epoch. Right: the average output of the discriminator on real and on generated images. Both panels are essentially flat after epoch 5, even though the samples keep improving." %}

After the first few epochs, nothing much happens in these plots. The losses settle and stay put. The discriminator assigns an average of about 0.76 to real digits and 0.24 to generated ones for the rest of training, never approaching the theoretical value of 0.5. Yet the samples in the previous figure improve visibly between epoch 5 and epoch 15. In supervised learning a falling loss means the model is getting better. In a GAN, each loss is measured against an opponent that is also changing, so a flat curve means the two players are keeping pace with each other, which says nothing about how good either of them is. This is the first practical lesson of GAN training: you have to look at samples, and ideally measure them.

### A crude measurement

One simple way to measure samples is to ask a separate classifier. We train a small convolutional network on real MNIST digits, then show it 10,000 generated images and see what it says. This is not the FID metric, which uses a much larger pretrained network and a more careful statistic [[7]](#references), only a coarse sanity check. It does, however, test two things we care about: whether the samples are recognizable, and whether all ten digits appear.

```python
# A GAN's loss does not say how good its samples are. One crude check is to ask a separate classifier,
# trained on real digits, what it sees in the generated ones. This is not FID, only a coarse sanity check.
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x):
        return self.net(x)

torch.manual_seed(0)
clf = SmallCNN().to(device)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
y_train_d = y_train.to(device)
for epoch in range(3):
    perm = torch.randperm(len(x_train), device=device)
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]
        loss = F.cross_entropy(clf(x_train[idx]), y_train_d[idx])
        opt.zero_grad(); loss.backward(); opt.step()
clf.eval()
with torch.no_grad():
    test_acc = (clf(x_test.to(device)).argmax(1).cpu() == y_test).float().mean().item()
    G.eval()
    gen = torch.cat([G(torch.randn(1000, latent_dim, device=device)) for _ in range(10)])   # 10,000 samples
    probs = F.softmax(clf(gen), dim=1).cpu()
pred = probs.argmax(1)
hist = torch.bincount(pred, minlength=10).float() / len(pred)
print(f"classifier accuracy on real test digits: {100 * test_acc:.1f}%")
print("share of generated samples assigned to each digit (%):", [round(100 * h, 1) for h in hist.tolist()])
with torch.no_grad():
    real_probs = F.softmax(clf(x_test[:10000].to(device)), dim=1).cpu()
for name, p in (("real test digits", real_probs), ("generated samples", probs)):
    conf = p.max(1).values
    print(f"{name:18s}: mean classifier confidence {conf.mean():.3f}, confidence below 0.9 for {100 * (conf < 0.9).float().mean():.1f}%")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(range(10), 100 * hist.numpy(), color="#08519c")
ax.axhline(10, color="gray", ls="--", lw=1)
ax.set_xticks(range(10)); ax.set_xlabel("digit the classifier assigns"); ax.set_ylabel("% of generated samples")
fig.tight_layout(); fig.savefig("figures/gan_class_hist.png", dpi=170, bbox_inches="tight")
```

```text
classifier accuracy on real test digits: 98.6%
share of generated samples assigned to each digit (%): [14.4, 7.9, 9.5, 7.5, 8.9, 12.4, 7.6, 12.9, 5.9, 12.8]
real test digits  : mean classifier confidence 0.987, confidence below 0.9 for 3.6%
generated samples : mean classifier confidence 0.871, confidence below 0.9 for 37.0%
```

{% include figure.liquid path="assets/img/blog/gan/gan_class_hist.png" class="img-fluid rounded z-depth-1" zoomable=true caption="How the classifier labels 10,000 generated images. The dashed line marks 10%, the share each digit would have if the generator matched the real class balance." %}

The classifier reads 98.6% of real test digits correctly and is confident about them: its average confidence is 0.987, and only 3.6% of real images get a confidence below 0.9. On the generated images the average confidence falls to 0.871, and 37% of them are below 0.9. Roughly one generated sample in three is ambiguous or malformed enough that a classifier trained on real digits hesitates. That agrees with what the sample grid shows to the eye.

The class balance is also skewed. If the generator matched the data, each digit would make up about 10% of the samples. Instead the shares range from 5.9% for the digit 8 up to 14.4% for the digit 0. Take this with some caution, because the classifier's own mistakes on sloppy samples feed into the count: a malformed 8 may be read as a 0 or a 5. It is an indication that the generator does not reproduce the class proportions of the data, and not a precise measurement of which digits it neglects. Notice also that the GAN was never told about classes. Nothing in the objective asks for balanced coverage, which is part of why the problem exists.

## Walking through the latent space

Because the generator is a smooth function of its noise input, nearby noise vectors give similar images. We can see that by drawing two noise vectors and decoding the points on the straight line between them.

```python
# Walk through the latent space: a straight line between two noise vectors.
torch.manual_seed(3)
rows = []
with torch.no_grad():
    for _ in range(4):
        z0, z1 = torch.randn(1, latent_dim, device=device), torch.randn(1, latent_dim, device=device)
        ts = torch.linspace(0, 1, 10, device=device).view(-1, 1)
        rows.append(G((1 - ts) * z0 + ts * z1).cpu())
fig, axes = plt.subplots(4, 10, figsize=(10, 4.2))
for r in range(4):
    for c in range(10):
        axes[r, c].imshow(rows[r][c, 0], cmap="gray", vmin=-1, vmax=1); axes[r, c].axis("off")
fig.tight_layout(pad=0.2); fig.savefig("figures/gan_interpolation.png", dpi=170, bbox_inches="tight")
print("saved")
```

{% include figure.liquid path="assets/img/blog/gan/gan_interpolation.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Each row decodes ten points along a straight line between two random noise vectors, from the left image to the right image. Digits change continuously from one to another, passing through plausible intermediate shapes." %}

Each row transforms one image into another by passing through intermediate shapes, which are themselves plausible-looking digits. This is evidence that the generator has not simply memorized a set of training images: it has learned something like a continuous space of digit shapes. It is evidence of a modest kind. It does not prove that the generator has captured the data distribution, and it does not rule out subtler memorization, which would need other tests, such as comparing each sample with its nearest training image.

## Why GANs are hard to train

The experiments above show most of the standard difficulties, and the research literature has spent years on each of them.

**Mode collapse and dropping.** Nothing in the objective explicitly rewards the generator for covering the whole distribution. If it finds a few outputs that the current discriminator accepts, it can sit there until the discriminator catches up, and then jump to another. The 25-mode experiment shows the effect on a toy.

**Non-convergence.** Gradient descent on a minimax game does not behave like gradient descent on a loss. The two players can cycle indefinitely, and theory shows that even simple examples have this problem without special treatment [[3]](#references).

**Vanishing gradients.** If the discriminator becomes too good, it separates real and fake with high confidence, and the generator's gradients vanish. The non-saturating loss helps. There is also a more fundamental problem: if the real and generated distributions live on different low-dimensional surfaces, which is typical for images, a perfect discriminator exists, and the JS divergence is constant, so it provides no useful gradient at all. This observation motivated the Wasserstein GAN [[4]](#references), which replaces the JS divergence with a distance that stays informative in that case, and its gradient-penalty variant [[5]](#references).

**No good loss to monitor.** As seen above, the training losses do not track sample quality. Evaluation has to use samples, through metrics such as FID [[7]](#references), which has its own limitations.

Common stabilizers include spectral normalization of the discriminator [[6]](#references), gradient penalties on the discriminator, label smoothing, two-timescale updates with different learning rates for the two networks, and careful architecture choices. Large-scale GANs that produce high-resolution images use all of these and many more, and still require a lot of tuning.

## Common implementation errors

- **Forgetting `detach()` on the fake batch** when training the discriminator. Without it the discriminator's loss is also backpropagated into the generator, which wastes computation, and if the generator's gradients are not cleared before its own update, the discriminator's loss leaks into the generator's step.
- **Mismatched value ranges.** Real images in $$[0, 1]$$ and a generator ending in `tanh` (range $$[-1, 1]$$) let the discriminator tell them apart by range alone.
- **Wrong target for the generator step.** The generator's loss uses the label *real* for fake samples. Using the label *fake* with a flipped sign gives the saturating loss, and some implementations mix the two by accident.
- **Applying the sigmoid twice.** If the discriminator ends in a sigmoid, use plain binary cross-entropy; if it outputs logits, use the `WithLogits` loss. Doing both squashes the signal.
- **Batch normalization statistics.** Feeding real and fake samples in separate batches gives batch-norm layers different statistics in each case, which changes what the discriminator learns. It is a standard choice, but it is a choice, and mixing the batches is another.
- **Judging by loss curves.** As the experiment showed, a flat loss is not a sign of failure or success. Look at samples every few epochs.
- **A single seed.** The 25-mode experiment gave very different results with different seeds. Report several.

## Where GANs stand today

For a few years GANs were the best image generators, and they remain important in several places. Their sampling is fast, since producing an image takes one pass through the generator, where diffusion models need many steps, as diffusion models, which generate an image by many small denoising steps, show. Adversarial losses are widely used as one component of other systems, for example in training the image autoencoders that latent diffusion models rely on [[8]](#references), and in some distillation methods that compress slow generators into fast ones. For generating images from scratch, diffusion models have taken over in most large-scale work, in part because they are easier to train and scale, and cover the data distribution more evenly. The cost of the adversarial game is that it is hard to make it reliable, which is what the experiments above illustrate.

## References

1. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. "Generative Adversarial Nets." *NeurIPS*, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
2. A. Radford, L. Metz, and S. Chintala. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." *ICLR*, 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
3. L. Mescheder, A. Geiger, and S. Nowozin. "Which Training Methods for GANs do actually Converge?" *ICML*, 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
4. M. Arjovsky, S. Chintala, and L. Bottou. "Wasserstein Generative Adversarial Networks." *ICML*, 2017. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
5. I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. "Improved Training of Wasserstein GANs." *NeurIPS*, 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
6. T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. "Spectral Normalization for Generative Adversarial Networks." *ICLR*, 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957)
7. M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS*, 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
8. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*, 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
