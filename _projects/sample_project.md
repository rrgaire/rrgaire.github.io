---
# SAMPLE / TEMPLATE — this project is NOT published (see `published: false`).
# To add a project:
#   1. Copy this file to _projects/<short-name>.md
#   2. Edit the front matter below and delete `published: false`
#   3. Preview at http://localhost:8080/projects/
#   (The Projects page is hidden from the navbar: set `nav: true` in _pages/projects.md to show it.)
layout: page
title: Sample project
description: One-line summary shown on the project card
img: assets/img/projects/sample.jpg # card thumbnail (optional)
importance: 1 # lower number = shown first
category: work # must match one of display_categories in _pages/projects.md (work, fun)
related_publications: false # true adds a "References" list of papers cited with {% cite key %} on this page
published: false # <- delete this line to publish
# redirect: https://github.com/rrgaire/your-repo  # optional: card links straight to a URL instead
---

Short overview of the project: the problem, your approach, and the result.

## Highlights

- What you built
- Key result / metric
- Links: [code](https://github.com/rrgaire) · [paper](https://example.com)

## Figures

Put images in `assets/img/projects/`, then:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/projects/sample.jpg" title="example" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">A caption under the figure.</div>
