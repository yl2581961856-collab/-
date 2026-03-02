# Personal Knowledge Site

This repository powers a personal website for iterating on my knowledge system and recording real‑world lessons learned. The content focuses on continuous updates, practical experiments, and distilled takeaways.

## Tech Stack

- Eleventy (11ty)
- Markdown + Nunjucks

## Local Development

```bash
npm install
npm run dev
```

Open:

- http://localhost:8080

## Build

```bash
npm run build
```

Output is generated in `dist/`.

## Content Structure

- `src/posts/` for blog posts
- `learning/`, `projects/`, `explorations/` for topic notes and archives

### Post Front Matter

```yaml
---
title: "Title"
description: "Short summary"
date: 2026-03-02
category: "Theory"
tags: ["Pretrain", "Fine-tune"]
---
```

## Purpose

- Evolve a personal knowledge system
- Track practical experiments and pitfalls
- Keep long‑term technical notes organized
