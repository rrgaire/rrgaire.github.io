# rrgaire.github.io

Personal academic website of Rebati Gaire — live at <https://rrgaire.github.io>.

Built with [Jekyll](https://jekyllrb.com/) on the [al-folio](https://github.com/alshedivat/al-folio) theme (MIT License, see `LICENSE`).

## Run locally

Requires only [Docker](https://docs.docker.com/get-docker/). Ruby, Jekyll, and all gems live inside the image.

```bash
docker compose up -d
```

The first run builds the image from `Dockerfile` (a few minutes). Then open <http://localhost:8080>. Stop with `docker compose down`.

Versions are pinned: Ruby in `Dockerfile` (same version in `.github/workflows/deploy.yml`) and every gem in `Gemfile.lock`.

## Where things live

| What          | Where                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| About page    | `_pages/about.md`                                                                                     |
| CV            | `assets/json/resume.json` ([JSON Resume](https://jsonresume.org/schema) format), PDF in `assets/pdf/` |
| Publications  | `_bibliography/papers.bib`                                                                            |
| News          | `_news/`                                                                                              |
| Blog posts    | `_posts/` (see `_posts/2026-11-19-sample-post.md` for a template)                                     |
| Projects      | `_projects/` (see `_projects/sample_project.md` for a template)                                       |
| Site settings | `_config.yml`                                                                                         |

Theme documentation: `INSTALL.md`, `CUSTOMIZE.md`, `FAQ.md`.

## Deploy

Pushing to `master` triggers the **Deploy site** GitHub Action, which builds the site and publishes it to the `gh-pages` branch.
