---
title: Welcome
description: Exploring AI, ML & science
slug: welcome
date: 2026-01-20
image: cover.jpg
categories:
    - Reflections
tags:
keywords:
    - AI
    - Machine Learning
    - Graph Theory
    - LLM fine-tuning
    - Data poisoning
    - Quantum computing
    - LoRA
weight: 10
toc: true
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

This blog is a technical notebook to explore topics I’m curious about. Everything here—code, data, and analysis—is hosted on GitHub to ensure others can inspect, reproduce, or build upon my work.

---

## Current interests & ideas

* **AI & Machine Learning:** LLM fine-tuning, data poisoning, and predictive modeling.
* **Graph Theory & Network Science:** Analyzing political networks (parliamentary voting behavior) and modeling epidemic spreads.
* **Quantum Computing & Cryptography:** Exploring the fundamentals of quantum information—such as running single-qubit circuits via the Felis framework and Alice & Bob’s cat qubit technology.
* **Decision Systems & Strategy:** Game theory applied to sports (tennis) or board games, and reinforcement learning.
* **Reflections on AI & Sovereignty:** Analyzing how Large Language Models shape public opinion and impact democratic stability. My focus is on the dominance of a few tech giants and how states use these tools to project influence and challenge national autonomy.

---

## The hardware constraint

I run my experiments on a **2017 MacBook Air (8GB RAM, Dual-Core i5)** without paid cloud services. 
This constraint encourages me to focus on small LLMs and lightweight techniques like LoRA. I prioritize deep understanding and efficiency over chasing massive scale.

> **Update:** To fine-tune a 7B-parameter LLM with QLoRA, I used a free T4 (16 GB) on Google Colab - my machine's 1.5 GB of VRAM makes it a non-starter for anything above a few hundred million parameters.

---

## Closing Note

This is a space for curiosity and open-source collaboration. Everything shared here is open for you to reuse, adapt, or critique. If a project sparks your interest, feel free to dive in and share your findings.



> Photo by [Pawel Czerwinski](https://unsplash.com/@pawel_czerwinski) on [Unsplash](https://unsplash.com/)