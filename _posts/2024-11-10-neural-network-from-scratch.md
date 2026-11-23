---
layout: post
title: "A Neural Network from Scratch: Forward Pass, Backpropagation, and Learning"
date: 2024-11-10 09:00:00-0600
description: Build and train a small multilayer perceptron in NumPy, derive every gradient by hand, check them numerically, and evaluate on a held-out test set.
tags: neural-networks backpropagation numpy tutorial
categories: tutorials
toc:
  sidebar: left
related_posts: false
---

Deep learning frameworks compute gradients for us, which makes it easy to train a model without seeing what
actually happens inside it. Writing a small network by hand is still one of the best ways to understand the
mechanics: what a forward pass computes, where each gradient comes from, and why training can quietly go
wrong. In this tutorial we build a two-layer multilayer perceptron (MLP) using only NumPy, derive its gradients,
verify them numerically, and train it on a small public dataset with a separate test set.

Everything below is one runnable script, split into sections. Copy the code blocks in order into a file such as
`mlp_from_scratch.py` and run it with `python mlp_from_scratch.py`. The results shown were produced with
Python 3.12, NumPy 2.1.3, and scikit-learn 1.5.2. scikit-learn is used **only** to load the dataset; the model,
loss, and backpropagation are written explicitly in NumPy.

```bash
pip install numpy==2.1.3 scikit-learn==1.5.2
```

## The dataset and preprocessing

We use the handwritten digits dataset that ships with scikit-learn: 1,797 grayscale images of size
$$8 \times 8$$, each labeled with a digit from 0 to 9. It is a copy of the test set of the UCI Optical Recognition
of Handwritten Digits dataset [[4]](#references). Each image is flattened into a vector of $$D = 64$$ pixel values.

Two details matter more than they first appear:

1. **Hold out a test set before doing anything else.** We shuffle once with a fixed seed and keep 20% of the
   examples (359 images) for testing. The remaining 1,438 images are used for training.
2. **Fit preprocessing on the training set only.** We standardize each pixel using the mean and standard
   deviation of the *training* images and apply the same transformation to the test images. Computing these
   statistics on all of the data would leak information from the test set into training.

```python
import numpy as np
from sklearn.datasets import load_digits  # used only to load the data

SEED = 0
rng = np.random.default_rng(SEED)
```

```python
digits = load_digits()                      # 1,797 grayscale 8x8 images, labels 0-9
X_all = digits.data.astype(np.float64)      # (1797, 64): one flattened image per row
y_all = digits.target                       # (1797,): integer class labels

# Shuffle once with a fixed seed, then hold out 20% as a test set.
perm = rng.permutation(len(X_all))
n_test = int(0.2 * len(X_all))
test_idx, train_idx = perm[:n_test], perm[n_test:]
X_train, y_train = X_all[train_idx], y_all[train_idx]   # (1438, 64), (1438,)
X_test, y_test = X_all[test_idx], y_all[test_idx]       # (359, 64), (359,)

# Standardize with statistics computed on the TRAINING set only.
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8          # some border pixels are always 0
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
```

## The model and its shapes

The network maps an input vector to a score (logit) for each of the $$C = 10$$ classes:

$$
\mathbf{x} \in \mathbb{R}^{D}
\;\xrightarrow{\;\text{linear}\;}\;
\mathbf{z}^{(1)} \in \mathbb{R}^{H}
\;\xrightarrow{\;\text{ReLU}\;}\;
\mathbf{a}^{(1)} \in \mathbb{R}^{H}
\;\xrightarrow{\;\text{linear}\;}\;
\mathbf{z}^{(2)} \in \mathbb{R}^{C}
\;\xrightarrow{\;\text{softmax}\;}\;
\mathbf{p} \in \mathbb{R}^{C}.
$$

We process a mini-batch of $$N$$ examples at once and store one example per **row**. With $$H = 64$$ hidden
units, the shapes are:

| Symbol | Meaning | Shape |
| :-- | :-- | :-- |
| $$X$$ | input batch | $$N \times D$$ |
| $$W_1,\ \mathbf{b}_1$$ | first-layer weights and bias | $$D \times H,\ H$$ |
| $$Z_1,\ A_1$$ | hidden pre-activations and activations | $$N \times H$$ |
| $$W_2,\ \mathbf{b}_2$$ | second-layer weights and bias | $$H \times C,\ C$$ |
| $$Z_2$$ | logits | $$N \times C$$ |
| $$P$$ | predicted class probabilities | $$N \times C$$ |
| $$Y$$ | one-hot labels | $$N \times C$$ |

Writing the shape next to every line of code is a cheap habit that prevents many bugs.

We initialize the ReLU layer with He initialization [[3]](#references), which draws weights with variance
$$2/D$$ so that activations neither vanish nor explode at the start of training. Biases start at zero.
Weights must **not** start at zero: every hidden unit would then compute the same function and receive the same
gradient, and they would never become different from each other.

```python
def init_params(D, H, C, rng):
    """He initialization for the ReLU layer, small random weights for the output."""
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / D), size=(D, H)),   # (D, H)
        "b1": np.zeros(H),                                     # (H,)
        "W2": rng.normal(0.0, np.sqrt(1.0 / H), size=(H, C)),   # (H, C)
        "b2": np.zeros(C),                                     # (C,)
    }
```

## Forward propagation

The forward pass is three matrix operations:

$$
Z_1 = X W_1 + \mathbf{1}\mathbf{b}_1^\top, \qquad
A_1 = \max(Z_1, 0), \qquad
Z_2 = A_1 W_2 + \mathbf{1}\mathbf{b}_2^\top,
$$

where $$\mathbf{1}$$ is a column of $$N$$ ones, so the bias is added to every row. In NumPy this is plain
broadcasting. The ReLU is applied element-wise. We keep $$X$$, $$Z_1$$, and $$A_1$$ in a cache because the
backward pass needs them.

```python
def forward(params, X):
    """X: (N, D) -> logits Z2: (N, C). Returns the cache needed for backprop."""
    Z1 = X @ params["W1"] + params["b1"]    # (N, H)
    A1 = np.maximum(Z1, 0.0)                # (N, H)  ReLU
    Z2 = A1 @ params["W2"] + params["b2"]   # (N, C)  logits
    cache = (X, Z1, A1)
    return Z2, cache
```

## Softmax and cross-entropy, computed stably

Softmax turns the logits of example $$i$$ into probabilities:

$$
P_{ij} = \frac{\exp(Z_{ij})}{\sum_{k=1}^{C} \exp(Z_{ik})}.
$$

The cross-entropy loss is the average negative log-probability assigned to the correct class $$y_i$$:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log P_{i, y_i}.
$$

Computed literally, this formula is fragile. A logit of 1000 makes $$\exp(1000)$$ overflow to infinity, and a
very negative logit makes a probability underflow to zero, so $$\log 0 = -\infty$$. Two observations fix both
problems:

1. Softmax does not change if we subtract the same constant from every logit of an example. Subtracting the
   row maximum $$m_i = \max_k Z_{ik}$$ makes the largest exponent $$\exp(0) = 1$$, so nothing overflows.
2. We never need the probabilities themselves to compute the loss, only their logarithms:

$$
\log P_{ij} = (Z_{ij} - m_i) - \log \sum_{k=1}^{C} \exp(Z_{ik} - m_i).
$$

This "log-sum-exp" form is what the code computes. It returns the probabilities as well, because the
backward pass uses them.

```python
def softmax_cross_entropy(Z, y):
    """Z: (N, C) logits, y: (N,) integer labels.
    Returns the mean cross-entropy loss and the probabilities P: (N, C)."""
    Z_shift = Z - Z.max(axis=1, keepdims=True)            # stability: largest logit is 0
    log_sum_exp = np.log(np.exp(Z_shift).sum(axis=1, keepdims=True))
    log_P = Z_shift - log_sum_exp                         # log-softmax, (N, C)
    N = Z.shape[0]
    loss = -log_P[np.arange(N), y].mean()
    return loss, np.exp(log_P)
```

## Deriving the gradients

Backpropagation [[1]](#references) is the chain rule applied in reverse order of the forward pass. We want the
gradient of $$L$$ with respect to every parameter. We write $$G_2 = \partial L / \partial Z_2$$,
$$G_1 = \partial L / \partial Z_1$$, and so on. Every gradient has the same shape as the quantity it is taken
with respect to, which gives a useful check on each step.

**Softmax and cross-entropy together.** For a single example with logits $$\mathbf{z}$$ and true class
$$y$$, the loss is

$$
\ell = -z_y + \log \sum_{k} \exp(z_k).
$$

Differentiating with respect to $$z_j$$:

$$
\frac{\partial \ell}{\partial z_j}
= -\mathbb{1}[j = y] + \frac{\exp(z_j)}{\sum_k \exp(z_k)}
= p_j - y_j,
$$

where $$y_j$$ is the one-hot label. Averaging over the batch adds a factor $$1/N$$:

$$
G_2 = \frac{\partial L}{\partial Z_2} = \frac{1}{N}\,(P - Y) \qquad (N \times C).
$$

This simple result is the reason softmax and cross-entropy are almost always implemented together.

**Second linear layer.** Element-wise, $$Z_{2,ic} = \sum_h A_{1,ih} W_{2,hc} + b_{2,c}$$. Each weight
$$W_{2,hc}$$ affects the logit of class $$c$$ for every example $$i$$, so its gradient sums over the batch:

$$
\frac{\partial L}{\partial W_{2,hc}} = \sum_{i} A_{1,ih}\, G_{2,ic}
\quad\Longrightarrow\quad
\frac{\partial L}{\partial W_2} = A_1^\top G_2 \qquad (H \times C).
$$

By the same argument, the bias gradient sums $$G_2$$ over the batch, and the gradient flowing back into the
hidden activations uses the transpose of $$W_2$$:

$$
\frac{\partial L}{\partial \mathbf{b}_2} = \sum_{i} G_{2,i:} \quad (C), \qquad
\frac{\partial L}{\partial A_1} = G_2 W_2^\top \quad (N \times H).
$$

**ReLU.** The derivative of $$\max(z, 0)$$ is 1 where $$z > 0$$ and 0 where $$z < 0$$. (At exactly
$$z = 0$$ we use 0, a standard convention.) The gradient passes through only where the unit was active:

$$
G_1 = \frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{1}[Z_1 > 0]
\qquad (N \times H),
$$

where $$\odot$$ is element-wise multiplication.

**First linear layer.** This has the same structure as the second layer, with $$X$$ as the input:

$$
\frac{\partial L}{\partial W_1} = X^\top G_1 \quad (D \times H), \qquad
\frac{\partial L}{\partial \mathbf{b}_1} = \sum_{i} G_{1,i:} \quad (H).
$$

The code follows these equations line by line:

```python
def backward(params, cache, P, y):
    """Gradients of the mean cross-entropy loss with respect to every parameter."""
    X, Z1, A1 = cache
    N = X.shape[0]

    dZ2 = P.copy()                          # (N, C)
    dZ2[np.arange(N), y] -= 1.0             # P - Y (one-hot Y)
    dZ2 /= N                                # mean over the batch

    dW2 = A1.T @ dZ2                        # (H, C)
    db2 = dZ2.sum(axis=0)                   # (C,)
    dA1 = dZ2 @ params["W2"].T              # (N, H)
    dZ1 = dA1 * (Z1 > 0)                    # (N, H)  ReLU derivative
    dW1 = X.T @ dZ1                         # (D, H)
    db1 = dZ1.sum(axis=0)                   # (H,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

## Checking the gradients numerically

A derivation can contain a mistake that still produces a program that runs and even learns a little. Before
training, we compare the analytic gradients with a numerical estimate. For each parameter entry
$$\theta$$, the central difference

$$
\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon}
$$

has an error proportional to $$\varepsilon^2$$. We compute it for every entry on a tiny random problem in
64-bit floating point and report the relative error
$$\lVert g_{\text{num}} - g \rVert / (\lVert g_{\text{num}} \rVert + \lVert g \rVert)$$.

```python
def gradient_check(seed=1, N=5, D=4, H=6, C=3, eps=1e-6):
    """Compare analytic gradients with central finite differences on a tiny problem."""
    r = np.random.default_rng(seed)
    params = init_params(D, H, C, r)
    X = r.normal(size=(N, D))
    y = r.integers(0, C, size=N)

    Z, cache = forward(params, X)
    _, P = softmax_cross_entropy(Z, y)
    grads = backward(params, cache, P, y)

    worst = 0.0
    for name, theta in params.items():
        num = np.zeros_like(theta)
        for i in np.ndindex(theta.shape):
            old = theta[i]
            theta[i] = old + eps
            loss_plus, _ = softmax_cross_entropy(forward(params, X)[0], y)
            theta[i] = old - eps
            loss_minus, _ = softmax_cross_entropy(forward(params, X)[0], y)
            theta[i] = old
            num[i] = (loss_plus - loss_minus) / (2 * eps)
        rel = np.linalg.norm(num - grads[name]) / max(
            np.linalg.norm(num) + np.linalg.norm(grads[name]), 1e-12)
        print(f"{name}: relative error = {rel:.2e}")
        worst = max(worst, rel)
    return worst

assert gradient_check() < 1e-6
```

Output:

```text
W1: relative error = 8.56e-10
b1: relative error = 3.18e-10
W2: relative error = 5.48e-10
b2: relative error = 2.39e-10
```

Relative errors around $$10^{-9}$$ mean the analytic and numerical gradients agree to many digits. An error in
the derivation usually produces values far larger, often close to 1. One caveat: if a hidden pre-activation
lies within $$\varepsilon$$ of zero, the ReLU kink can make a single entry disagree. On a tiny random problem
this is rare, and rerunning with another seed confirms whether it is the cause.

## Learning with gradient descent

Training repeats one update for every mini-batch. Each parameter moves a small step in the direction that
decreases the loss on that batch:

$$
\theta \leftarrow \theta - \eta\, \frac{\partial L_{\text{batch}}}{\partial \theta},
$$

with learning rate $$\eta = 0.1$$, mini-batches of 64 examples reshuffled every epoch, and 30 passes
(epochs) over the training set. After selected epochs we report the loss and accuracy on the full training
set and on the held-out test set.

```python
def evaluate(params, X, y):
    Z, _ = forward(params, X)
    loss, P = softmax_cross_entropy(Z, y)
    accuracy = (P.argmax(axis=1) == y).mean()
    return loss, accuracy

D, H, C = X_train.shape[1], 64, 10
params = init_params(D, H, C, rng)
learning_rate, batch_size, epochs = 0.1, 64, 30

history = []                                         # (epoch, train loss, train acc, test loss, test acc)
for epoch in range(1, epochs + 1):
    order = rng.permutation(len(X_train))            # new mini-batch order each epoch
    for start in range(0, len(X_train), batch_size):
        batch = order[start:start + batch_size]
        Z, cache = forward(params, X_train[batch])
        _, P = softmax_cross_entropy(Z, y_train[batch])
        grads = backward(params, cache, P, y_train[batch])
        for name in params:                          # gradient descent step
            params[name] -= learning_rate * grads[name]

    tr_loss, tr_acc = evaluate(params, X_train, y_train)
    te_loss, te_acc = evaluate(params, X_test, y_test)
    history.append((epoch, tr_loss, tr_acc, te_loss, te_acc))
    if epoch in (1, 5, 10, 20, 30):
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f}, acc {tr_acc:.3f} "
              f"| test loss {te_loss:.4f}, acc {te_acc:.3f}")
```

```text
epoch  1 | train loss 0.8163, acc 0.821 | test loss 0.8149, acc 0.819
epoch  5 | train loss 0.2119, acc 0.964 | test loss 0.2201, acc 0.964
epoch 10 | train loss 0.1084, acc 0.983 | test loss 0.1334, acc 0.972
epoch 20 | train loss 0.0522, acc 0.994 | test loss 0.0866, acc 0.972
epoch 30 | train loss 0.0320, acc 0.997 | test loss 0.0710, acc 0.983
```

Printing a handful of epochs is fine for a quick check, but a curve shows more. The same history, plotted:

```python
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

ep, tr_l, tr_a, te_l, te_a = np.array(history).T
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
ax1.plot(ep, tr_l, color="#08519c", lw=2, label="training")
ax1.plot(ep, te_l, color="#d95f02", lw=2, label="test")
ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy loss"); ax1.set_yscale("log"); ax1.legend(frameon=False)
ax2.plot(ep, 100 * tr_a, color="#08519c", lw=2, label="training")
ax2.plot(ep, 100 * te_a, color="#d95f02", lw=2, label="test")
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy (%)"); ax2.set_ylim(80, 100.5); ax2.legend(frameon=False)
fig.tight_layout(); fig.savefig("nn_curves.png", dpi=170)
```

{% include figure.liquid path="assets/img/blog/neural-network/nn_curves.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Loss (left, log scale) and accuracy (right) per epoch. Training loss keeps falling; the test loss follows it down and then flattens, and the gap between the two is the generalization gap." %}

The curves have the usual shape. Loss drops fastest in the first few epochs, when the weights are still far from anything useful, and then improves slowly. The test loss follows the training loss closely for about the first five epochs, then the two separate: the training loss keeps falling while the test loss flattens out above it. That widening gap is the generalization gap in picture form.

Numbers summarize a model and pictures let you check it. Here are the first 24 test images with the model's predictions, and a confusion matrix over the whole test set.

```python
Z_test, _ = forward(params, X_test)
pred = Z_test.argmax(axis=1)

fig, axes = plt.subplots(3, 8, figsize=(10, 4.2))
for ax, i in zip(axes.ravel(), range(24)):
    ax.imshow(digits.data[test_idx[i]].reshape(8, 8), cmap="gray_r")
    wrong = pred[i] != y_test[i]
    ax.set_title(f"{pred[i]}" + (f" (true {y_test[i]})" if wrong else ""), fontsize=9,
                 color="#b30000" if wrong else "black")
    ax.axis("off")
fig.tight_layout(); fig.savefig("nn_predictions.png", dpi=170)

confusion = np.zeros((10, 10), dtype=int)
for t, q in zip(y_test, pred):
    confusion[t, q] += 1
fig, ax = plt.subplots(figsize=(5, 4.4))
ax.imshow(confusion, cmap="Blues")
for t in range(10):
    for q in range(10):
        if confusion[t, q]:
            ax.text(q, t, confusion[t, q], ha="center", va="center", fontsize=8,
                    color="white" if confusion[t, q] > confusion.max() / 2 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("predicted digit"); ax.set_ylabel("true digit")
fig.tight_layout(); fig.savefig("nn_confusion.png", dpi=170)
print("misclassified test images:", int((pred != y_test).sum()), "of", len(y_test))
```

```text
misclassified test images: 6 of 359
```

{% include figure.liquid path="assets/img/blog/neural-network/nn_predictions.png" class="img-fluid rounded z-depth-1" zoomable=true caption="The first 24 test images and the model's predictions. Red titles would mark mistakes, but all of these first 24 happen to be correct. Each image is only 8 by 8 pixels, so some digits are ambiguous even to a person." %}

{% include figure.liquid path="assets/img/blog/neural-network/nn_confusion.png" class="img-fluid rounded z-depth-1" zoomable=true caption="Confusion matrix on the 359 test images. Rows are the true digit and columns the prediction, so everything off the diagonal is a mistake." %}


## Training loss and generalization

The training loss falls steadily, from 0.816 after the first epoch to 0.032 after 30. That shows the
optimization works: gradient descent finds parameters that fit the training images very well.

What we actually care about is performance on images the model has never seen. That is why the test set was
held out from the start and never used for fitting, preprocessing, or choosing settings. The test loss also
decreases, to 0.071, and the final test accuracy is 98.3% (353 of 359 images). The gap between training loss
(0.032) and test loss (0.071) is the **generalization gap**: the model fits its training data more closely
than new data. Here the gap is small. If we kept training a much larger network without regularization, the
training loss would keep shrinking while the test loss would eventually stop improving or rise. That is
overfitting.

Two cautions about reading these numbers. First, with 359 test images, one image is worth about 0.28
percentage points of accuracy, so small differences between runs or settings are not meaningful on their own.
Second, these results come from a single split and a single seed. If you start changing the learning rate or
hidden size, carve a **validation set** out of the training data and use it for those decisions. Reusing the
test set to pick settings would make its accuracy an optimistic estimate.

## Common implementation errors

- **Unstable softmax.** Exponentiating raw logits overflows, and taking the log of probabilities that underflow
  to zero gives infinities. Subtract the row maximum and compute log-probabilities directly.
- **Wrong averaging.** If the loss is a mean over the batch, the gradient needs exactly one factor of
  $$1/N$$. Dropping it, or applying it twice, silently changes the effective learning rate. The gradient check
  catches this because it compares against the loss you actually defined.
- **Broadcasting and shape mistakes.** Labels of shape `(N, 1)` instead of `(N,)`, or a bias gradient summed
  over the wrong axis, can produce arrays that broadcast without an error but contain the wrong numbers.
  Assert or comment the expected shapes.
- **Data leakage.** Fitting the standardization (or any preprocessing) on the full dataset before splitting
  lets test information influence training. Split first, then fit on the training set only.
- **Symmetric initialization.** Initializing all weights to zero or to the same value means every hidden unit
  receives the same gradient and they stay identical. Use small random values with an appropriate scale.
- **Learning rate too large.** If the loss grows or becomes `nan`, reduce the learning rate before suspecting
  the gradients. If the gradient check passes, the problem is usually the optimization settings.
- **Evaluating on the training set only.** High training accuracy says the model can fit the data, not that it
  generalizes. Always report results on held-out data.

## Where to go next

The same structure scales to deeper networks: each layer needs a forward function, a backward function that
maps the upstream gradient to gradients for its inputs and parameters, and a gradient check. Natural next steps
are adding a validation set and early stopping, L2 weight decay (which adds $$\lambda W$$ to each weight
gradient), and momentum or Adam in place of plain gradient descent. Chapter 6 of Goodfellow, Bengio, and
Courville [[2]](#references) covers these feedforward-network topics in depth.

## References

1. D. E. Rumelhart, G. E. Hinton, and R. J. Williams. "Learning representations by back-propagating errors."
   *Nature*, 323:533–536, 1986.
2. I. Goodfellow, Y. Bengio, and A. Courville. *Deep Learning*, Chapter 6: Deep Feedforward Networks.
   MIT Press, 2016. [deeplearningbook.org](https://www.deeplearningbook.org/)
3. K. He, X. Zhang, S. Ren, and J. Sun. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification." *ICCV*, 2015.
4. E. Alpaydin and C. Kaynak. "Optical Recognition of Handwritten Digits." UCI Machine Learning Repository,
   1998. [doi:10.24432/C50P49](https://doi.org/10.24432/C50P49)
