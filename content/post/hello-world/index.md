---
title: Working Notes
description: Exploring AI, Machine Learning, and Graphs under real-world constraints
slug: welcome
date: 2026-01-20 00:00:00+0000
image: cover.jpg
categories:
    - Reflections
tags:
    - Artificial Intelligence
    - Machine Learning
    - Graph Theory
    - Security
    - Reproducible Research
weight: 10
toc: false
---


<style>
  .article-page .main-article {
    width: 95% !important;
    max-width: 1000px !important; 
    margin: 0 auto !important;
    float: none !important;
  }

  .article-content img:not(table img), 
  .article-content canvas, 
  .article-content svg,
  .article-content iframe {
    max-width: 100% !important;
    height: auto !important;
    display: block;
    margin: 1em auto;
  }

  .article-content table {
    display: block;
    width: 100% !important;
    overflow-x: auto;
    border-collapse: collapse;
    margin-bottom: 1.5em;
  }

  .article-content table img {
    max-width: 280px !important;
    height: auto !important;
    display: inline-block;
  }

  @media screen and (max-width: 600px) {
    .article-content table img {
      max-width: 180px !important;
    }
  }

  @media (min-width: 1024px) {
    .main-container {
      grid-template-columns: 220px 1fr !important; 
      gap: 40px !important;
      max-width: 1400px !important;
      margin: 0 auto !important;
    }

    .article-page .main-article {
      max-width: 1100px !important; 
    }
  }
</style>

---

# Working Notes

**Exploring AI, Machine Learning, and Graphs — with Limited Resources and Serious Questions**

This blog serves as a technical and scientific notebook. The objective is to explore ideas through concrete experiments and to improve understanding by confronting theory with practice. Content is built around hands-on experimentation, literature review, and reproducible analysis. When possible, code and experimental artifacts are made available on GitHub to enable inspection and reproduction. Results may be partial or inconclusive; documenting limitations and failures is considered as important as reporting successful outcomes.

This blog sits at the intersection of **machine learning, mathematics, security, and real-world systems**.

### Artificial Intelligence & Machine Learning
- Fine-tuning small language models
- Evaluating models beyond raw accuracy
- Predictive machine learning inspired by academic data challenges

### Security & Robustness
- Data poisoning and backdoor attacks in LLMs
- Failure modes under adversarial or corrupted inputs

### Graph Theory & Network Analysis
- Political networks inferred from parliamentary voting behavior
- Detection of clusters, hubs, bridges, and outliers
- Measuring polarization, cooperation, and influence

### Mathematics & Decision Systems
- Game theory and strategic interactions
- Reinforcement learning in simplified environments

### Reflections on AI
- What large language models are good at — and what they are not
- How AI tools can either dull thinking or sharpen it
- Ethical, cognitive, and societal implications of automation

## Working Under Strong Constraints

All experiments published here are conducted under **strict and explicit constraints**.

My main machine is a **2017 MacBook Air**:
- Intel Core i5 (dual-core, 1.8 GHz)
- 8 GB RAM
- No paid cloud services

These constraints force:
- Small LLM models
- Quantization and lightweight fine-tuning (e.g. LoRA)
- Efficient ML algorithms
- A clear focus on understanding rather than scale

## Methodology

Whenever possible, articles will follow a research-inspired structure:

1. **Question or Hypothesis**  
2. **Context & Related Work**  
3. **Methodology**  
4. **Experiments**  
5. **Results**  
6. **Limitations**  
7. **Discussion**

Code, data, and notebooks will be shared on GitHub when feasible, with a focus on reproducibility.

## AI as a Tool for Thought

Artificial intelligence — especially large language models — sits at a critical crossroads.

Used poorly, it can:
- Encourage shallow understanding
- Replace thinking with pattern matching

Used carefully, it can:
- Accelerate exploration
- Support hypothesis generation
- Help test ideas faster

## Closing Note

This blog is a space to explore ideas, run experiments, and nurture curiosity. Here, the focus is on learning through hands-on testing, observation, and reflection.
All projects shared here — code, datasets, and reflections — are open source. Feel free to reuse, adapt, or build upon them. If something sparks your interest, dive in and share your own findings! Science thrives on collaboration and dialogue, and your contributions are more than welcome.


> Photo by [Pawel Czerwinski](https://unsplash.com/@pawel_czerwinski) on [Unsplash](https://unsplash.com/)