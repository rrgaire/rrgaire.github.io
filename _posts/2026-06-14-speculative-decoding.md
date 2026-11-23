---
layout: post
title: "Speculative Decoding: Faster Language Model Inference Without Changing the Output"
date: 2026-06-14 10:00:00-0600
description: A first-principles explanation of speculative decoding, with a proof that it samples from exactly the same distribution as the large model, a derivation of the expected speedup, and a NumPy implementation that tests both claims.
tags: speculative-decoding inference llm tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

A large language model writes text one token at a time. Each token needs a full pass through the network, and the next token cannot start until the previous one is known. This is the main reason that generating a long answer feels slow even on expensive hardware, and it is the reason that a lot of current research is about making each of those steps cheaper or fewer.

Speculative decoding takes a different route. It does not shrink the model, change its weights, or approximate its output. A small, fast model guesses several tokens ahead, and the large model checks all the guesses at once. Tokens that pass the check are kept. The procedure can be designed so that the final text has *exactly* the same distribution as if the large model had generated it alone. The idea was introduced independently by Leviathan, Kalman, and Matias [[1]](#references) and by Chen et al. [[2]](#references).

This post explains why it works, proves the correctness claim, derives how much speedup to expect, and tests both claims with a short NumPy program. If you have not seen how a language model produces tokens, [the post on how LLMs work]({% post_url 2025-09-07-how-large-language-models-work %}) covers the background. You only need basic probability here.

## Why decoding is slow

Generating a token with a model of $$P$$ parameters takes about $$2P$$ floating-point operations, one multiply and one add per weight. That is not much work for a modern accelerator. The trouble is that every one of those weights has to be read from memory to do it. For a 7-billion-parameter model stored in 16-bit precision, that is about 13.5 GB of data moved per token. A GPU with roughly 2 TB/s of memory bandwidth needs at least 6.7 ms just to stream the weights once, however fast its arithmetic units are, which caps generation near 150 tokens per second for a single sequence. The arithmetic intensity, the number of operations per byte moved, is about 1 for this workload, while the hardware can sustain on the order of 150. The chip spends most of its time waiting for memory [[3]](#references).

That observation points to the opportunity. If the model is waiting on memory rather than on arithmetic, then asking it to process *several* tokens in a single pass costs very little extra. The weights are read once either way. The only question is where to find several tokens worth processing when we do not yet know what the next one is.

## Checking is cheaper than writing

A transformer with a causal mask computes, in one forward pass, the next-token distribution at every position of the input. When we train the model, we use exactly this property: feed in a whole sentence, and get a prediction for each token from the tokens before it. At generation time we normally waste it, because we only have one new position to look at.

Suppose that someone hands us a candidate continuation of $$\gamma$$ tokens. We can feed the context plus the candidate into the large model in a single pass and obtain its distributions at all $$\gamma + 1$$ relevant positions. In the memory-bound regime described above, that costs roughly as much as generating one token, because the weights are read from memory only once either way. The candidate can come from several places. A smaller model that shares the large model's vocabulary is the standard source, and it is the one we use here. (Proposers without a probability model, such as lookup of repeated n-grams, can be handled by treating their proposal as a distribution with all its mass on one token.) The remaining problem is to decide which of the proposed tokens to keep, in a way that does not distort the output.

## The algorithm

Call the large model the **target** and the small one the **draft**. Let $$p(\cdot)$$ be the target's next-token distribution and $$q(\cdot)$$ the draft's, at whatever position we are discussing. One round of speculative decoding does the following.

1. **Draft.** The draft model samples $$\gamma$$ tokens autoregressively, $$x_1, \dots, x_\gamma$$, recording its distribution $$q_i$$ at each step.
2. **Score.** The target model runs once on the context plus the draft tokens and returns its distributions $$p_1, \dots, p_{\gamma+1}$$. Here $$p_i$$ is the target's distribution for position $$i$$ given the context and the first $$i-1$$ draft tokens.
3. **Verify.** Go through the draft tokens from left to right. Accept $$x_i$$ with probability $$\min\!\big(1,\; p_i(x_i)/q_i(x_i)\big)$$.
4. **Repair.** At the first rejection, discard that token and everything after it. Sample a replacement from the *residual distribution* $$p'_i$$, defined just below, and end the round.
5. **Bonus.** If all $$\gamma$$ draft tokens are accepted, sample one more token from $$p_{\gamma+1}$$. This is free, since the target already computed that distribution.

The residual distribution keeps only the probability mass by which the target exceeds the draft, and rescales it to sum to one:

$$
p'_i(x) = \frac{\max\big(0,\; p_i(x) - q_i(x)\big)}{\sum_{y} \max\big(0,\; p_i(y) - q_i(y)\big)}.
$$

Every round therefore emits between 1 and $$\gamma + 1$$ tokens and costs one target pass plus $$\gamma$$ draft passes. In the worst case we get one token per round, which is what ordinary decoding would have given us, plus the wasted cost of the draft.

A small example makes the rule concrete. Take a five-token vocabulary with

$$
p = (0.50,\; 0.25,\; 0.15,\; 0.07,\; 0.03), \qquad q = (0.30,\; 0.30,\; 0.20,\; 0.10,\; 0.10).
$$

The draft over-weights tokens 1 through 4 and under-weights token 0. If the draft proposes token 0, the acceptance probability is $$\min(1, 0.5/0.3) = 1$$, so it is always kept. If it proposes token 4, the acceptance probability is $$0.03/0.10 = 0.3$$, so it is usually rejected. The residual is $$\max(0, p - q) = (0.20, 0, 0, 0, 0)$$, which normalizes to $$(1, 0, 0, 0, 0)$$. So in this example, whenever a rejection happens, the replacement is token 0, the one token the draft did not propose often enough. The two corrections push in opposite directions and balance exactly, which the next section shows in general.

## Why the output has the target's distribution

Fix one position and condition on everything before it. Let $$\tilde{x}$$ denote the token that the procedure emits there, either an accepted draft token or the replacement. We will show that $$\Pr[\tilde{x} = x] = p(x)$$ for every token $$x$$.

There are two ways to emit $$x$$. The first is that the draft proposes $$x$$ and it is accepted:

$$
\Pr[\text{propose } x \text{ and accept}] = q(x)\,\min\!\Big(1, \frac{p(x)}{q(x)}\Big) = \min\big(p(x), q(x)\big).
$$

The second is that a rejection occurs and the residual distribution produces $$x$$. The total probability of a rejection, which we call $$\rho$$, is

$$
\rho = 1 - \sum_y \min\big(p(y), q(y)\big) = \sum_y \big(p(y) - \min(p(y), q(y))\big) = \sum_y \max\big(0, p(y) - q(y)\big).
$$

That is exactly the normalizing constant of the residual distribution, so $$\rho > 0$$ whenever a rejection can happen. (If $$p = q$$, no rejection ever occurs and the residual is never needed.) So

$$
\Pr[\text{reject and then emit } x] = \rho \cdot \frac{\max(0, p(x) - q(x))}{\rho} = \max\big(0, p(x) - q(x)\big).
$$

Adding the two cases,

$$
\Pr[\tilde{x} = x] = \min\big(p(x), q(x)\big) + \max\big(0, p(x) - q(x)\big) = p(x).
$$

The last equality holds because for any two numbers $$a$$ and $$b$$, $$\min(a, b) + \max(0, a - b) = a$$. This is the same acceptance-and-correction trick that appears in classical rejection sampling, adapted so that a rejected proposal is replaced by a draw from the leftover mass instead of being retried.

The argument above is for a single position. For a whole sequence, apply it position by position: once a token has been emitted, the next step is the same one-position problem with a longer context. The first emitted token has distribution $$p(\cdot \mid \text{context})$$ by the result above. Given that token, whatever route produced it, the next emitted token is again handled by the same procedure with the longer context: if the first token was an accepted draft token, the round continues and tests the next draft token against $$p$$ for the extended context; if it was a replacement, the round ends and a new one starts from that extended context. Either way the second token has distribution $$p(\cdot \mid \text{context}, \text{first token})$$. By the chain rule of probability, the product of these conditionals is the target's distribution over the whole sequence [[1]](#references).

Two remarks on the scope of this guarantee. It holds for any draft distribution $$q$$, even a terrible one. A poor draft does not change what is generated, only how fast. The requirement is about which distributions enter the acceptance test. $$p$$ must be the distribution you want the output to follow, so if you sample with temperature, top-$$k$$, or nucleus filtering, apply those settings to the target's logits first. $$q$$ must be exactly the distribution the draft token was sampled from, whatever settings produced it. Applying the same filters to the draft is a good idea because it raises the acceptance rate, but exactness does not depend on it.

## How many tokens per round

The expected number of tokens per round determines the speedup, so let us compute it. Define the acceptance rate

$$
\alpha = \mathbb{E}\Big[\sum_x \min\big(p(x), q(x)\big)\Big],
$$

which is the probability that a draft token is accepted. At a single position the acceptance probability equals $$1 - \mathrm{TV}(p, q)$$, where TV is the total variation distance between the two distributions, so $$\alpha$$ is one minus the average TV distance. In the example above, $$\alpha = 0.3 + 0.25 + 0.15 + 0.07 + 0.03 = 0.8$$.

If we treat the acceptance of each draft token as an independent event with probability $$\alpha$$, the first $$i$$ draft tokens are all accepted with probability $$\alpha^i$$. A round always emits one token (the replacement or the bonus) plus one for each accepted draft token before the first rejection, so

$$
\mathbb{E}[\text{tokens per round}] = 1 + \sum_{i=1}^{\gamma} \alpha^i = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}.
$$

This matches Eq. 1 in Leviathan et al. [[1]](#references). Real acceptance rates vary from position to position, so the formula is an approximation for real models and exact when the acceptance probability is the same everywhere.

Now let the draft model be cheaper to run than the target by a factor $$c$$, meaning that one draft step costs $$c$$ times as much as one target step. A round costs $$\gamma c + 1$$ target-step equivalents, assuming that scoring $$\gamma + 1$$ positions costs the same as scoring one, which is the memory-bound assumption from the first section. The expected speedup over ordinary decoding is therefore

$$
\text{speedup} = \frac{1 - \alpha^{\gamma + 1}}{(1 - \alpha)(\gamma c + 1)},
$$

which is the wall-clock improvement factor of Leviathan et al. [[1]](#references). The formula shows the trade-off in choosing $$\gamma$$. A longer draft raises the numerator, but with diminishing returns because $$\alpha^{\gamma+1}$$ shrinks geometrically (slowly when $$\alpha$$ is close to 1), while the denominator grows linearly. There is an optimal $$\gamma$$ for each pair $$(\alpha, c)$$.

## An implementation to test the claims

We will implement the algorithm for toy models whose next-token distributions we can write down exactly, so that we can compare the output of speculative decoding with the true distribution rather than with another sample. The "models" are functions from a context to a probability vector. The same code would work with real models if `p_fn` and `q_fn` called them.

The core function follows the five steps above. The `buggy` flag switches the repair step to a common mistake, resampling from $$p$$ instead of the residual, so that we can see what an incorrect implementation looks like.

```python
import numpy as np

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def sample(p, rng):
    """Draw one index from the probability vector p."""
    return min(int(np.searchsorted(np.cumsum(p), rng.random())), len(p) - 1)

def residual(p, q):
    """The distribution norm(max(0, p - q)) used after a rejection."""
    r = np.maximum(p - q, 0.0)
    return r / r.sum()

def speculative_round(prefix, p_fn, q_fn, gamma, rng, buggy=False):
    """One draft-and-verify round. Returns the tokens emitted (between 1 and gamma + 1)."""
    # 1. The draft model proposes gamma tokens, one at a time.
    ctx, drafts, q_dists = list(prefix), [], []
    for _ in range(gamma):
        q = q_fn(ctx)
        x = sample(q, rng)
        q_dists.append(q)
        drafts.append(x)
        ctx.append(x)
    # 2. The target model scores all gamma + 1 positions (one forward pass in a real system).
    p_dists = [p_fn(list(prefix) + drafts[:i]) for i in range(gamma + 1)]
    # 3. Verify the draft tokens from left to right.
    out = []
    for i, x in enumerate(drafts):
        p, q = p_dists[i], q_dists[i]
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x)                                   # accept the draft token
        else:
            fix = p if buggy else residual(p, q)            # buggy variant: resample from p itself
            out.append(sample(fix, rng))                    # replacement token
            return out                                      # stop at the first rejection
    out.append(sample(p_dists[-1], rng))                    # every draft accepted: bonus token
    return out
```

Two small drivers: one for speculative generation of a fixed number of tokens, and one for ordinary sampling from the target to serve as the reference.

```python
def generate(prefix, length, p_fn, q_fn, gamma, rng, buggy=False):
    """Speculative generation of `length` tokens. Returns (tokens, number of target passes)."""
    seq, rounds = list(prefix), 0
    while len(seq) - len(prefix) < length:
        seq += speculative_round(seq, p_fn, q_fn, gamma, rng, buggy)
        rounds += 1
    return seq[len(prefix):len(prefix) + length], rounds

def generate_plain(prefix, length, p_fn, rng):
    """Ordinary sampling from the target model, one token per step."""
    seq = list(prefix)
    for _ in range(length):
        seq.append(sample(p_fn(seq), rng))
    return seq[len(prefix):]
```

### Test 1: one position

First the single-position claim. We use the five-token $$p$$ and $$q$$ from the worked example, run one round with $$\gamma = 1$$ many times, and record the first token emitted. For comparison we also run the buggy variant, and compute its distribution analytically. Since a rejection then replaces the token with a sample from $$p$$, the output probability would be $$\min(p, q) + \rho\, p$$ with $$\rho = 1 - \alpha$$.

```python
rng = np.random.default_rng(0)
p = np.array([0.50, 0.25, 0.15, 0.07, 0.03])      # target distribution
q = np.array([0.30, 0.30, 0.20, 0.10, 0.10])      # draft distribution
alpha = np.minimum(p, q).sum()                    # acceptance probability
p_fn, q_fn = (lambda ctx: p), (lambda ctx: q)     # same distributions at every position

print("alpha =", round(alpha, 3))
print("residual after a rejection:", np.round(residual(p, q), 3))
N = 200_000
print("target                  ", p)
freqs = {}
for name, buggy in [("correct", False), ("resample from p", True)]:
    counts = np.zeros(len(p))
    for _ in range(N):
        counts[speculative_round([], p_fn, q_fn, 1, rng, buggy)[0]] += 1
    freq = counts / N
    freqs[name] = freq
    print(f"{name:24s}", np.round(freq, 4), f" TV = {0.5 * np.abs(freq - p).sum():.4f}")
print("buggy, exact calculation ", np.round(np.minimum(p, q) + (1 - alpha) * p, 4))
```

```text
alpha = 0.8
residual after a rejection: [1. 0. 0. 0. 0.]
target                   [0.5  0.25 0.15 0.07 0.03]
correct                  [0.5004 0.2496 0.15   0.07   0.03  ]  TV = 0.0004
resample from p          [0.3998 0.2987 0.1814 0.0841 0.036 ]  TV = 0.1002
buggy, exact calculation  [0.4   0.3   0.18  0.084 0.036]
```

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 3.7))
x, w = np.arange(len(p)), 0.2
ax.bar(x - 1.5 * w, p, w, color="#08519c", label="target p")
ax.bar(x - 0.5 * w, q, w, color="#bdbdbd", label="draft q")
ax.bar(x + 0.5 * w, freqs["correct"], w, color="#1b9e77", label="output of speculative decoding")
ax.bar(x + 1.5 * w, freqs["resample from p"], w, color="#d95f02", label="output of the buggy variant")
ax.set_xticks(x); ax.set_xticklabels([f"token {i}" for i in x]); ax.set_ylabel("probability")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("figures/spec_distribution.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/speculative-decoding/spec_distribution.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The target distribution p (blue), the draft distribution q (gray), and the empirical output distributions of correct speculative decoding (green) and the buggy variant (orange). The green bars sit on top of the blue ones; the orange bars do not." %}

The correct implementation reproduces $$p$$ to within sampling noise (the total variation distance is 0.0004 here; with 200,000 draws, distances of a few thousandths would also be unremarkable). The buggy variant is off by 0.10 in total variation, and its empirical frequencies agree with the exact calculation, so the discrepancy is real and not noise. The error is also easy to miss in casual testing: the buggy output still looks like a sensible distribution, and it is closer to $$p$$ than $$q$$ is. Only comparing against the exact target reveals it.

### Test 2: tokens per round

Next the formula for the expected number of tokens per round. With the same fixed $$p$$ and $$q$$, the acceptance probability is identical at every position, which is the case where the formula is exact.

```python
print("gamma   simulated   formula")
simulated = {}
for gamma in [1, 2, 3, 4, 6, 8]:
    emitted = [len(speculative_round([], p_fn, q_fn, gamma, rng)) for _ in range(50_000)]
    simulated[gamma] = np.mean(emitted)
    formula = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    print(f"{gamma:5d}   {np.mean(emitted):9.3f}   {formula:7.3f}")
```

```text
gamma   simulated   formula
    1       1.800     1.800
    2       2.438     2.440
    3       2.956     2.952
    4       3.346     3.362
    6       3.965     3.951
    8       4.342     4.329
```

```python
gs = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(gs, (1 - alpha ** (gs + 1)) / (1 - alpha), color="#08519c", lw=2, label="formula")
ax.plot(list(simulated), list(simulated.values()), "o", color="#d95f02", ms=7, label="simulation")
ax.set_xlabel("draft tokens per round, gamma"); ax.set_ylabel("tokens emitted per round"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/spec_tokens_per_round.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/speculative-decoding/spec_tokens_per_round.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Average number of tokens emitted per round as the number of draft tokens grows, for an acceptance rate of 0.8. The curve is the formula and the dots are simulations. Returns diminish as gamma grows." %}

The simulated averages follow the formula closely. The small differences are sampling error from 50,000 rounds per row. Note how the gains flatten: going from $$\gamma = 4$$ to $$\gamma = 8$$ doubles the draft work but adds only about one extra token per round.

### Test 3: whole sequences with context-dependent models

The last test is the strongest. Here the models condition on the previous token, so the distribution at each position depends on what came before, and the draft is a noisy version of the target. We generate sequences of three tokens, which gives 64 possible outcomes. Because the target is a small Markov chain, the exact probability of each sequence is a product of table entries. We compare the empirical frequencies of speculative decoding, ordinary sampling, and the buggy variant against those exact probabilities with a chi-square statistic, which should land near the number of degrees of freedom (63) if the samples really follow the exact distribution.

```python
import itertools
rng = np.random.default_rng(1)
V, L, N = 4, 3, 100_000
logits = rng.normal(size=(V, V)) * 1.5
P = softmax(logits)                                   # target: row = previous token
Q = softmax(logits + rng.normal(size=(V, V)) * 0.8)   # draft: related to the target but imperfect
p_fn, q_fn = (lambda ctx: P[ctx[-1]]), (lambda ctx: Q[ctx[-1]])
start = [0]

seqs = list(itertools.product(range(V), repeat=L))
index = {s: i for i, s in enumerate(seqs)}
exact = np.array([P[start[0], s[0]] * P[s[0], s[1]] * P[s[1], s[2]] for s in seqs])

def empirical(make_sequence):
    counts = np.zeros(len(seqs))
    for _ in range(N):
        counts[index[tuple(make_sequence())]] += 1
    return counts / N

runs = {
    "ordinary sampling": empirical(lambda: generate_plain(start, L, p_fn, rng)),
    "speculative":       empirical(lambda: generate(start, L, p_fn, q_fn, 4, rng)[0]),
    "buggy variant":     empirical(lambda: generate(start, L, p_fn, q_fn, 4, rng, buggy=True)[0]),
}
dof = len(seqs) - 1
print(f"{len(seqs)} possible sequences; chi-square should be near {dof} (give or take {(2 * dof) ** 0.5:.0f})")
for name, freq in runs.items():
    chi2 = N * ((freq - exact) ** 2 / exact).sum()
    print(f"{name:18s} chi-square = {chi2:8.1f}   TV distance = {0.5 * np.abs(freq - exact).sum():.4f}")
```

```text
64 possible sequences; chi-square should be near 63 (give or take 11)
ordinary sampling  chi-square =     63.9   TV distance = 0.0089
speculative        chi-square =     55.6   TV distance = 0.0073
buggy variant      chi-square =  12100.3   TV distance = 0.1319
```

Ordinary sampling and speculative decoding both land within the expected range of 63 (give or take about 11), so neither can be distinguished from the exact target distribution at this sample size. (One of the 64 sequences has an expected count of about 3.4, a little under the usual rule of thumb of 5 for the chi-square approximation, so treat the statistic as a rough guide. The gap to the buggy variant is far too large for that to matter.) The buggy variant gives a statistic of about 12,000. A test of this kind is worth keeping around when you build a speculative decoding system: it catches the mistakes that make output slightly wrong but still fluent.

This test supports the correctness argument but does not replace it. A chi-square test cannot prove equality, and with finite samples it can only fail to find a difference. The proof in the earlier section is what establishes exactness.

### What the speedup formula predicts

Finally, a table of the speedup formula for a few choices of acceptance rate $$\alpha$$ and relative draft cost $$c$$. These numbers come from the formula, not from timing a real system.

```python
def improvement(alpha, gamma, c):
    """Expected wall-clock speedup when one draft step costs c target steps."""
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))

gammas = [1, 2, 3, 4, 5, 6, 8, 10]
print("gamma            " + "".join(f"{g:6d}" for g in gammas))
for c in (0.02, 0.1):
    for a in (0.6, 0.8, 0.9):
        print(f"c={c:<4} alpha={a}  " + "".join(f"{improvement(a, g, c):6.2f}" for g in gammas))
```

```text
gamma                 1     2     3     4     5     6     8    10
c=0.02 alpha=0.6    1.57  1.88  2.05  2.13  2.17  2.17  2.13  2.08
c=0.02 alpha=0.8    1.76  2.35  2.78  3.11  3.35  3.53  3.73  3.81
c=0.02 alpha=0.9    1.86  2.61  3.24  3.79  4.26  4.66  5.28  5.72
c=0.1  alpha=0.6    1.45  1.63  1.67  1.65  1.59  1.52  1.37  1.25
c=0.1  alpha=0.8    1.64  2.03  2.27  2.40  2.46  2.47  2.40  2.29
c=0.1  alpha=0.9    1.73  2.26  2.65  2.93  3.12  3.26  3.40  3.43
```

```python
fig, ax = plt.subplots(figsize=(7.5, 4))
gs = np.arange(1, 13)
for c, ls in ((0.02, "-"), (0.1, "--")):
    for a_, color in ((0.6, "#d95f02"), (0.8, "#08519c"), (0.9, "#1b9e77")):
        ax.plot(gs, [improvement(a_, g, c) for g in gs], ls, color=color, lw=2, label=f"alpha = {a_}, c = {c}")
ax.axhline(1.0, color="gray", lw=1)
ax.set_xlabel("draft tokens per round, gamma"); ax.set_ylabel("expected speedup over ordinary decoding")
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig("figures/spec_speedup.png", dpi=170, bbox_inches="tight")
```

{% include figure.liquid path="assets/img/blog/speculative-decoding/spec_speedup.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Predicted speedup over ordinary decoding as a function of the number of draft tokens, for three acceptance rates (colors) and two relative draft costs (solid: c = 0.02, dashed: c = 0.1). These curves come from the formula, not from timing a real system." %}

With a cheap draft ($$c = 0.02$$) and a high acceptance rate ($$\alpha = 0.9$$), the model predicts more than a 5× gain at $$\gamma = 8$$ to $$10$$. With a draft that costs a tenth of the target and an acceptance rate of 0.6, the best case is under 1.7×, and the gain shrinks to 1.25× if $$\gamma$$ is pushed to 10. The two quantities that matter, $$\alpha$$ and $$c$$, pull against each other in practice: a bigger draft model agrees with the target more often but costs more to run.

## What the model leaves out

The formula describes the ideal case, and real deployments differ from it in several ways.

- **Extra computation.** Speculative decoding performs *more* total arithmetic than ordinary decoding, since rejected draft tokens are computed and thrown away, and the target scores positions that may not be used. It trades spare compute for lower latency. When a server is already compute-bound, for instance when it batches many requests together, the benefit shrinks or disappears.
- **The memory-bound assumption.** Scoring $$\gamma + 1$$ positions at the price of one only holds while the pass is limited by memory traffic. For long drafts, large batches, or hardware with high bandwidth relative to its arithmetic throughput, the verification pass starts to cost more.
- **Acceptance varies.** Text that is easy to predict, such as boilerplate, code syntax, or repetition of earlier content, gets high acceptance rates. Open-ended text gets lower ones. The measured speedup therefore depends on the task, not just on the model pair.
- **The draft must share the vocabulary.** The acceptance ratio compares $$p(x)$$ with $$q(x)$$ for the same token $$x$$, so the two models need the same tokenizer. A small model from the same family is the usual choice.
- **Overheads.** Real systems must also manage the key-value caches of both models and roll back the cache entries for rejected tokens, and these implementation costs reduce the ideal gain.

## Variations and related work

The accept-and-correct rule is general, and much of the research since the original papers concerns where the draft tokens come from. Not every method below keeps the exact-distribution guarantee in every mode, so check each one's sampling procedure. EAGLE and SpecInfer are designed to preserve the target distribution. Medusa's default acceptance rule for sampled generation is a relaxed criterion that trades exactness for acceptance rate, and exactness holds there for greedy decoding or when using a rejection-sampling variant. Some systems train extra prediction heads on the target model itself so that no separate draft model is needed, as in Medusa [[4]](#references). Others draft in the target's feature space, as in EAGLE [[5]](#references), which improves the acceptance rate. Tree-based approaches such as SpecInfer [[6]](#references) verify several candidate continuations at once by arranging them in a tree and using a suitable attention mask. Earlier ideas such as blockwise parallel decoding [[7]](#references) were already exploring parallel verification of multiple tokens, though without the exact-distribution guarantee in this form.

Two further points are worth knowing. With greedy decoding (temperature zero) the target distribution puts all of its mass on one token, so the rule reduces to a simple check: keep a draft token if and only if it is the target's most likely token. And the guarantee is about distributions, not about identical bits. Batched or reshaped computation can change floating-point rounding, so in rare near-ties a greedy run might differ in a token or two from a non-speculative run on the same hardware, even though the procedure is exact in exact arithmetic.

## Common implementation errors

- **Resampling from $$p$$ instead of the residual.** The most common mistake, and the one tested above. The output looks fine and is wrong.
- **Forgetting to renormalize the residual.** $$\max(0, p - q)$$ sums to the rejection probability $$\rho$$, not to 1.
- **A missed optimization: dropping the bonus token.** When all drafts are accepted, the target has already computed $$p_{\gamma+1}$$. Ignoring it is still correct, but it wastes one free token per fully accepted round.
- **Wrong distributions in the ratio.** The ratio must use the processed target distribution you actually want to sample from (after temperature, top-$$k$$, or top-$$p$$) as $$p$$, and the exact distribution the draft token was sampled from as $$q$$. Using unprocessed target probabilities, or a $$q$$ that differs from the one used for sampling, breaks the guarantee.
- **Cache bookkeeping.** After a rejection, the key-value entries for the discarded tokens must be removed from both models, or later tokens will attend to text that was never emitted.

## Where to go next

If you want to try this on real models, the Hugging Face `transformers` library supports assisted generation with a separate draft model, and serving frameworks such as vLLM include speculative decoding. The two original papers [[1]](#references) [[2]](#references) are both short and readable, and the proof above follows the first. Chen et al. also analyze the algorithm for a fixed number of draft tokens and report results on a 70-billion-parameter model.

## References

1. Y. Leviathan, M. Kalman, and Y. Matias. "Fast Inference from Transformers via Speculative Decoding." *ICML*, 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
2. C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper. "Accelerating Large Language Model Decoding with Speculative Sampling." 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
3. S. Williams, A. Waterman, and D. Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4):65–76, 2009.
4. T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." *ICML*, 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
5. Y. Li, F. Wei, C. Zhang, and H. Zhang. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." *ICML*, 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)
6. X. Miao et al. "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification." *ASPLOS*, 2024. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
7. M. Stern, N. Shazeer, and J. Uszkoreit. "Blockwise Parallel Decoding for Deep Autoregressive Models." *NeurIPS*, 2018. [arXiv:1811.03115](https://arxiv.org/abs/1811.03115)
