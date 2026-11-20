---
# SAMPLE / TEMPLATE — this post is NOT published (see `published: false`).
# To write a new post:
#   1. Copy this file to _posts/YYYY-MM-DD-your-title.md  (the date in the filename is required)
#   2. Edit the front matter below and delete `published: false`
#   3. Preview at http://localhost:8080/blog/ (docker compose up -d)
layout: post
title: Sample post title
date: 2026-11-19 10:00:00-0600
description: One-line summary shown on the blog list page
tags: machine-learning computer-vision # space-separated
categories: research # optional
published: false # <- delete this line to publish
featured: false # true pins the post at the top of /blog/
toc:
  sidebar: left # table of contents on the left; remove these 2 lines to disable
giscus_comments: false # comments (needs giscus configured in _config.yml)
related_posts: false
related_publications: true # adds a "References" list of papers cited with {% cite key %} below
---

Opening paragraph. Regular **Markdown** works everywhere: _italics_, **bold**, `inline code`,
and [links](https://example.com).

## A section heading

Lists:

- first point
- second point

> Blockquotes look like this.

## Images

Put images in `assets/img/` (for example `assets/img/blog/my-figure.png`), then:

{% include figure.liquid loading="eager" path="assets/img/blog/my-figure.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<div class="caption">A caption under the image.</div>

## Code

```python
import torch

x = torch.randn(1, 3, 224, 224)
print(x.shape)
```

## Math

Inline math: $$ E = mc^2 $$. Display math:

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{N} y_i \log p_\theta(x_i)
$$

## Tables

| Model    | Params | Top-1 |
| :------- | -----: | ----: |
| ResNet50 |    25M | 76.1% |
| ViT-B/16 |    86M | 81.8% |

## Citing your own papers

Use a key from `_bibliography/papers.bib`: {% cite gaire2025carn %}

More features (videos, charts, Jupyter notebooks, diagrams, tabs, and more) are shown in the
al-folio demo posts: https://github.com/alshedivat/al-folio/tree/main/_posts
