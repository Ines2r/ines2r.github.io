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
/* 2. LE TABLEAU (Général) */
  .article-content table {
    width: 100% !important;
    display: table !important;
    border-collapse: collapse;
    margin-bottom: 2em;
  }

  /* Par défaut, on laisse les colonnes s'ajuster (pour le tableau des couleurs) */
  .article-content td {
    vertical-align: top !important; 
    padding: 10px !important;
    border-bottom: 1px solid #eee;
  }

  /* 3. EXCEPTION POUR LES GRAPHES (Législatures) */
  /* Si le tableau a 3 colonnes, on force les largeurs égales pour vos graphes */
  .article-content table tr th:first-child:nth-last-child(3),
  .article-content table tr th:first-child:nth-last-child(3) ~ th,
  .article-content table tr td:first-child:nth-last-child(3),
  .article-content table tr td:first-child:nth-last-child(3) ~ td {
    width: 33.33% !important;
  }

  /* 4. GESTION INTELLIGENTE DES IMAGES */
  .article-content table img {
    height: auto !important;
    display: block;
  }

  /* Si l'image est un "badge" de couleur (Shields.io), on la garde petite */
  .article-content table img[src*="shields.io"] {
    width: 90px !important; /* Taille fixe pour vos carrés de couleur */
    display: inline-block;
  }

  /* Si c'est un graphique (pas un badge), il prend toute la place de sa colonne */
  .article-content table img:not([src*="shields.io"]) {
    width: 100% !important;
    max-width: none !important;
  }
</style>

---

# Working Notes

**Exploring AI, Machine Learning, and Graphs — with Limited Resources**

This blog is my technical notebook to explore topics I’m curious about and bridge the gap between theory and practice through hands-on experiments.

Everything here—code, data, and analysis—is hosted on **GitHub** to ensure others can inspect, reproduce, or build upon my work.

---

### Current Interests & Research Ideas

These are topics I am thinking about:

* **AI & Machine Learning:** LLM fine-tuning, data poisoning, and predictive modeling (academic challenges/real-world data).
* **Graph Theory & Network Science:** Analyzing political networks (parliamentary voting behavior) and modeling epidemic spreads.
* **Quantum Computing & Cryptography:** Exploring the fundamentals of quantum information—such as running single-qubit circuits via the Felis framework and Alice & Bob’s cat qubit technology.
* **Decision Systems & Strategy:** Game theory applied to sports (tennis) or board games, and reinforcement learning.
* **Reflections on AI & Sovereignty:** Analyzing how Large Language Models shape public opinion and impact democratic stability. I’m particularly interested in the dominance of a few tech giants and how states leverage these tools to project influence and challenge national autonomy.

---

### The "Low-Tech" Constraint

I run my experiments on a **2017 MacBook Air (8GB RAM, Dual-Core i5)** without paid cloud services. 

This constraint encourages me to focus on small LLMs and lightweight techniques like LoRA. For me, it’s about deep understanding and efficiency rather than massive scale.

---

## Closing Note

This is a space for curiosity and open-source collaboration. Everything shared here is open for you to reuse, adapt, or critique. If a project sparks your interest, feel free to dive in and share your findings.



> Photo by [Pawel Czerwinski](https://unsplash.com/@pawel_czerwinski) on [Unsplash](https://unsplash.com/)