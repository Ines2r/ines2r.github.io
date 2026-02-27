---
title: "Mapping the French National Assembly: A Network Analysis of voting behavior (2012–2024)"
description: "This analysis uses graph-based methods to map the relationships and voting proximities between Members of Parliament across three consecutive legislatures."
slug: networks-analysis
date: 2026-02-14 00:00:00+0000
image: Cover.png
math: true
categories:
    - Networks_Graphs
tags:
    - PCA
    - Graph Theory
    - Networks Analysis
    - Political Networks
weight: 2
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

# Networks-Analysis

This project uses Graph Theory to analyze voting patterns in the French National Assembly. By treating MPs as nodes and shared votes as edges, we reveal the structure of political life, moving beyond simple party labels.

**Keywords:** Quantitative political analysis; graphs; PCA; parliamentary votes; similarity analysis

---

## 1. Introduction and Political Context

### 1.1 Institutional Framework: The French National Assembly

The French National Assembly is the lower chamber of the French bicameral parliament. It is composed of **577 Members of Parliament** elected by a two-round single-member plurality voting system in geographically defined constituencies. Members of Parliament form **political groups** organized according to their electoral and ideological affinities.

For illustrative purposes, Figure 1 presents the composition of three consecutives legislatures by political group:

| 14th Legislature | 15th Legislature | 16th Legislature |
| :---: | :---: | :---: |
| <a href="L14_distribution.png" target="_blank"><img src="L14_distribution.png" style="height: 220px; cursor: zoom-in;"></a> | <a href="L15_distribution.png" target="_blank"><img src="L15_distribution.png" style="height: 220px; cursor: zoom-in;"></a> | <a href="L16_distribution.png" target="_blank"><img src="L16_distribution.png" style="height: 220px; cursor: zoom-in;"></a> |

**Figure 1:** Distribution of the 577 Members of Parliament by political group.

Each Member of Parliament (MP) in the National Assembly is affiliated with a specific political group. While these groups often correspond to a single political party, this is not always the case. A notable example is the Rassemblement National (RN) during the 15th legislature (2017–2022): although several MPs were members of this party, they did not form an official parliamentary group.

> We have more than 577 MPs because of resignations and replacements during the legislature.

### 1.2 Motivations and Research Questions

The objective of this project is to map the various political currents and their relative positioning by leveraging parliamentary voting data. Several questions naturally arise:

1. What is the true ideological landscape beyond group labels?
2. Does the historical left-right divide still exist?
3. How has this structure changed between two major legislatures?

Our approach:
- Treats each ballot vote as a **dimension in a political vector space**
- Uses **cosine distance** as a similarity metric
- Applies **spatialization** techniques (force-directed layout) and **principal component analysis** to visualize these data.

### 1.3 Overview of Main Political Forces and Ideological Positioning

To facilitate the interpretation of the spatialization graphs, the table below summarizes the main political groups, their associated colors in our study, and their core principles according to their official platforms.


| Color | Party/Group | Brief Description (Official Stance) | Source |
| :--- | :--- | :--- | :--- |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=e74c3c" style="width:20px !important; height:20px !important; display:inline;"> | **LFI / LFI-NUPES** | Focuses on ecological planning, wealth redistribution, and a constitutional shift to a 6th Republic. | [lafranceinsoumise.fr](https://lafranceinsoumise.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=2ecc71" style="width:20px !important; height:20px !important; display:inline;"> | **ECOLO** | Advocates for environmental sustainability, social-ecology, and biodiversity protection. | [lesecologistes.fr](https://lesecologistes.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=e84393" style="width:20px !important; height:20px !important; display:inline;"> | **SRC / SER / SOC** | Social-democratic model, defense of public services, and labor rights (Historical Socialist groups). | [parti-socialiste.fr](https://parti-socialiste.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=c0392b" style="width:20px !important; height:20px !important; display:inline;"> | **GDR** | Defense of the working class, social justice, and opposition to liberal economic policies. | [pcf.fr](https://pcf.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=f1c40f" style="width:20px !important; height:20px !important; display:inline;"> | **REN / LREM** | Supports economic competitiveness, full employment policies, and European integration. | [parti-renaissance.fr](https://parti-renaissance.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=e67e22" style="width:20px !important; height:20px !important; display:inline;"> | **DEM / RRDP** | Centrist approach, institutional balance, and education (includes center-left Radicals). | [mouvementdemocrate.fr](https://www.mouvementdemocrate.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=8e44ad" style="width:20px !important; height:20px !important; display:inline;"> | **HOR** | Focuses on long-term national stability, security, and supporting the presidential majority. | [horizonsleparti.fr](https://horizonsleparti.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=3498db" style="width:20px !important; height:20px !important; display:inline;"> | **UMP / LR** | Advocates for fiscal discipline, restoration of state authority, and economic liberalism. | [republicains.fr](https://republicains.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=2564a4" style="width:20px !important; height:20px !important; display:inline;"> | **RN** | Prioritizes national sovereignty, immigration control, and national priority policies. | [rassemblementnational.fr](https://rassemblementnational.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=d35400" style="width:20px !important; height:20px !important; display:inline;"> | **UDI / UAI** | Liberal-humanist and pro-European project centered on a social-market economy. | [parti-udi.fr](https://parti-udi.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=95a5a6" style="width:20px !important; height:20px !important; display:inline;"> | **LIOT** | An independent group focused on territorial interests, decentralization, and local governance. | [groupe-liot.fr](https://groupe-liot.fr) |
| <img src="https://img.shields.io/static/v1?label=&message=%20&color=bdc3c7" style="width:20px !important; height:20px !important; display:inline;"> | **NI** | Non-affiliated members who do not belong to any parliamentary group. | [assemblee-nationale.fr](https://www.assemblee-nationale.fr) |


> **Note:** Descriptions are synthesized from the "Manifesto" or "Our Values" sections of the parties' official websites to ensure alignment with their self-defined political identity.
---


## 2. Theoretical Framework and Methodology

### 2.1 Mathematical Foundations: Vector Representation of Votes

#### 2.1.1 The Space of Parliamentary Votes

Each Member of Parliament can be represented by a **vote vector** $\mathbf{v}_i \in \mathbb{R}^n$, where $n$ is the number of ballot votes analyzed. Formally:

$$
\mathbf{v}_i = (v_{i,1}, v_{i,2}, \ldots, v_{i,n})
$$


Each component $v_{i,j}$ encodes the MP's position on a specific vote $j$:
- $+1$ : In favor (Vote for the motion)
- $-1$ : Against (Vote against the motion)
- $0$ : Abstention (Present but did not take a position)
- $\text{NaN}$ : Absence (Not present during the session)


#### 2.1.2 Vote Matrix and Data Structure

After collecting votes via the NosDéputés.fr API, we construct a **vote matrix** $M \in \mathbb{R}^{m \times n}$:

- $m$ = number of MPs
- $n$ = number of ballot votes.

Each row corresponds to an MP's vote vector, and each column corresponds to a specific vote.

$$
M = \begin{pmatrix}
v_{1,1} & v_{1,2} & \cdots & v_{1,n} \\
v_{2,1} & v_{2,2} & \cdots & v_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
v_{m,1} & v_{m,2} & \cdots & v_{m,n}
\end{pmatrix}
$$


### 2.2 Handling Missing Data: The Parliamentary NaN Problem

**Major challenge:** The matrix $M$ contains missing values (NaN), corresponding to parliamentary absences. They reflect a latent variable: parliamentary engagement.

Two analytical frameworks are considered:

- **“Presence-Only” (Pearson, Agreement Ratio):** Focuses strictly on ideological alignment during shared presence. It is "agnostic" to the total volume of activity, but highly volatile when participation is low.
- **“Volume-Aware” (Cosine, Jaccard):** Integrates the level of activity into the geometry. By treating absence as a null component ($0$), it stabilizes the positions of inactive MPs by preventing them from reaching extreme similarity scores based on a single shared vote.

### 2.3 Four Similarity Metrics

#### 2.3.1 Cosine Similarity (Chosen Approach)

Cosine similarity measures the **directional alignment** between two vectors:

$$
S_{\cos}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|} = \frac{\sum_{j=1}^{n} A_j B_j}{\sqrt{\sum_{j=1}^{n} A_j^2} \sqrt{\sum_{j=1}^{n} B_j^2}}
$$

**Implementation:** NaN values are imputed as $0$.

**Justification:**
- **Directional Consistency:** $S_{\cos}$ captures the "political trajectory." Two MPs with identical voting ratios will point in the same direction, regardless of their total volume of votes.
- **Geometric Interpretation:** The denominator acts as a normalizer, while the numerator captures agreements (positive) and disagreements (negative).

**Properties:**
- Range: $S_{\cos} \in [-1, +1]$.
- Scale-invariant: Independent of the number of votes, provided the participation is sufficient to establish a stable direction.

#### 2.3.2 Pearson Correlation

$$
\rho(\mathbf{A}, \mathbf{B}) = \frac{\mathbb{E}[(\mathbf{A} - \bar{A})(\mathbf{B} - \bar{B})]}{\sigma_A \sigma_B}
$$

**NaN handling:** Pairwise deletion.

**Limitation:** Pearson suffers from **high sample bias** in sparse datasets. If two MPs coincide on only a few votes and agree, Pearson yields a perfect correlation ($+1.0$), creating "false positive" ideological alliances and over-inflating the importance of rare voters.

#### 2.3.3 Jaccard Similarity

$$
S_{\text{Jaccard}} = \frac{|\text{Agreements}|}{|P_A \cup P_B|}
$$

where $P_A$ and $P_B$ are the sets of votes where MPs A and B were present.

**Problem:** This metric is overly sensitive to participation gaps. The **union** in the denominator aggressively penalizes differences in activity levels, "diluting" the similarity toward zero even if the MPs agree on 100% of their shared presence.

#### 2.3.4 Weighted Agreement (Agreement Ratio)

$$
S_{\text{agreement}} = \frac{|\text{Agreements}|}{|P_A \cap P_B|}
$$

**Problem:** This often leads to **geometric degeneracy**. Since members of the same party follow strict voting instructions, their intersection of votes often results in $S = 1.0$. In a PCA or network graph, this causes entire political groups to **collapse into a single point**.

### 2.4 Synthesis: Choice of Metric

For this study, **we favor cosine similarity**. Unlike Pearson, it avoids over-inflating similarities based on tiny samples. Unlike the Agreement Ratio, it preserves the granularity of the Assembly by allowing subtle differences in participation and individual deviations to translate into distinct geometric coordinates.

---

## 3. Voting Network

### 3.1 Graph Construction via k-NN

Rather than creating a complete graph (potentially 150k+ edges), we use a **k-nearest neighbors topology**:

For each Member of Parliament $i$:
1. Compute $S_{\cos}(i, j)$ for all other MPs $j$
2. Retain the $k = 5$ neighbors with the highest similarity
3. Add a weighted edge $(i, j)$ with weight = $S_{\cos}(i, j)$

### 3.2 Layout Algorithm: Spring Model

To spatialize the graph in 2D, we apply the **Fruchterman–Reingold** algorithm (force-directed layout). All MPs are thrown randomly onto the plot, and the algorithm iteratively adjusts their positions until the system reaches equilibrium, driven by two competing forces:

$$
F_{\text{rep}}(i, j) = \frac{k}{d_{ij}}
$$
$$
F_{\text{attr}}(i, j) = \text{weight}_{ij} \cdot \left(-\frac{d_{ij}^2}{k}\right)
$$

where:
- $\text{weight}_{ij}$ = The Cosine Similarity between MP $i$ and MP $j$. The more they vote alike, the stronger the pull.
- $F_{\text{rep}}$ = repulsion (every node pushes others away to avoid clutter).
- $F_{\text{attr}}$ = attraction
- $d_{ij}$ = Euclidean distance between MP $i$ and MP $j$ on the 2D map.
- $k$: A balance parameter. It's tuned for visual harmony and does not affect the relative positions.

For Linked MPs (The Top 5 Neighbors): the distance $d_{ij}$ directly reflects the Cosine Similarity. If MP A and MP B have a weight of $0.95$, the attraction force $F_{\text{attr}}$ is very strong. They will be pulled until their Euclidean distance $d_{ij}$ is very small. In this case, small distance = high similarity.

For Non-Linked MPs (Everyone else): the distance $d_{ij}$ does not directly reflects their similarity. Instead, it reflects their relative position in the political ecosystem. The algorithm doesn't see a similarity of $0.05$ between a Far-Left MP and a Far-Right MP because they aren't in each other's Top 5. However, because the Far-Left MP is pulled to the "Left cluster" and the Far-Right MP is pulled to the "Right cluster," the Repulsion force ($F_{\text{rep}}$) and the chain of other connections will push them to opposite sides of the map. They end up far apart ($d_{ij}$ is large), not because the algorithm calculated their specific disagreement, but because they have no common friends to pull them together.


**Visual Interpretation:**
* **Political Clusters:** Groups with high voting discipline (e.g., LFI or Renaissance) naturally form dense, color-coded clouds.
* **Pivots and Bridges:** MPs who frequently vote across party lines are positioned between these clusters, acting as geometric "bridges."

| 14th Legislature (2012-2017) | 15th Legislature (2017-2022) | 16th Legislature (2022-2024) |
| :---: | :---: | :---: |
| <a href="L14_network_cosine.png" target="_blank"><img src="L14_network_cosine.png" style="height: 250px; width: auto; cursor: zoom-in;" alt="L14"></a> | <a href="L15_network_cosine.png" target="_blank"><img src="L15_network_cosine.png" style="height: 250px; width: auto; cursor: zoom-in;" alt="L15"></a> | <a href="L16_network_cosine.png" target="_blank"><img src="L16_network_cosine.png" style="height: 250px; width: auto; cursor: zoom-in;" alt="L16"></a> |

**Figure 2:** Graph of the 14th, 15th, and 16th legislatures using cosine similarity. Nodes are colored by political group. Note the increased fragmentation in the 16th legislature.


#### Observations:

- The decline of traditional pillars: Over the last decade, the Socialist and Republican parties have effectively moved from the center to the margins of the political landscape.
- Structural reshuffling: New parties have emerged, forming entirely new coalitions.
- Transversal behavior: Several MPs act as bridges between clusters. For instance, the Horizons group appears stretched: some members overlap with Renaissance, while others remain closer to the Republicans, reflecting a split in voting proximity.


## 4. Principal Component Analysis (PCA): Reduction and Visualization

We saw in section 2.1 that each MP is represented by a vote vector in a high-dimensional space ($\mathbb{R}^n$ where $n$ is the number of ballot votes). To visualize this $n$-dimensional voting space, we apply Principal Component Analysis (PCA). This dimensionality reduction technique projects the voting vectors onto a 2D plane (PC1 and PC2), preserving the maximum variance. This allow us to geographically map political distances: two deputies appearing close on the plot share a high proximity in their voting records.


### 4.1 Theoretical Foundations of PCA


Formally, let $\mathbf{M} \in \mathbb{R}^{m \times n}$ be the voting matrix (with $m$ MPs and $n$ votes). We first transform it into a standardized matrix $\mathbf{X}\_{\text{std}}$ where each element $x\_{i,j}$ is defined as:

$$x_{i,j} = \frac{m_{i,j} - \mu_j}{\sigma_j}$$

Where:
* **$m_{i,j}$**: The vote of MP $i$ for ballot $j$ (Abstention = 0).
* **$\mu_j$**: The **mean vote** for ballot $j$: $\mu_j = \frac{1}{m} \sum_{i=1}^{m} m_{i,j}$.
* **$\sigma_j$**: The **standard deviation** of ballot $j$: $\sigma_j = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (m_{i,j} - \mu_j)^2}$.

**Standardization Justification:** By centering each column and scaling to unit variance, we ensure that every vote has equal statistical weight. This prevents "landslide" votes (where almost everyone agrees) from drowning out more subtle, but politically significant, contested votes where the assembly is deeply divided.


#### 1. Finding the Principal Axes
PCA identifies the two principal axes $\mathbf{u}_1, \mathbf{u}_2$ that maximize the **explained variance**. In other words, it looks for the directions along which the MPs are the most spread out:

$$
\mathbf{u}_k = \arg\max_{\|\mathbf{u}\|=1} \text{Var}(\mathbf{X}_{\text{std}} \mathbf{u})
$$

* **PC1 (The Primary Cleavage):** The axis that captures the largest share of variance. In the French National Assembly, this should represent the **Government vs. Opposition** divide.
* **PC2 (The Secondary Nuance):** The axis perpendicular (orthogonal) to PC1 capturing the next largest source of variation (e.g., internal dissent or transverse issues).

#### 2. Geometric Projection
Each MP's standardized vector $\mathbf{x_i} \in \mathbb{R}^n$ is projected onto this plane to obtain their 2D coordinates $(z_{i,1}, z_{i,2})$:

$$
z_{i,1} = \mathbf{x_i} \cdot \mathbf{u_1}, \quad z_{i,2} = \mathbf{x}_i \cdot \mathbf{u}_2
$$

---

### 4.2 Interpretation of the PCA plots

1.  **Average behavior:** Because the data is centered via `StandardScaler`, the origin of the PCA plot represents the **mathematical average behavior** of the Assembly. 
2.  **Absenteeism:** Visually, the Assembly structure appears as several "branches" (each representing a political group) that radiate from a common junction. MPs with high absenteeism have vectors that lack the "magnitude" necessary to be projected far along the principal axes. Consequently, they naturally gravitate toward the convergence point of these branches.
3. **The 2D Projection Limit:** While "thin" clusters might suggest high voting discipline, this is a 2D projection. A cluster that appears compact may be significantly more dispersed along the higher-order principal components. While we can identify "bridge" individuals, visual dispersion should not be used as a definitive proxy for internal party cohesion.

| 15th Legislature (2017-2022) | 16th Legislature (2022-2024) |
| :---: | :---: |
| <a href="L15_pca_Global.png" target="_blank"><img src="L15_pca_Global.png" style="width: 100%; max-height: 500px; object-fit: contain; cursor: zoom-in;" alt="PCA L15"></a> | <a href="L16_pca_Global.png" target="_blank"><img src="L16_pca_Global.png" style="width: 100%; max-height: 500px; object-fit: contain; cursor: zoom-in;" alt="PCA L16"></a> |

**Figure 5:** Principal Component Analysis for all ballot

A reliable PCA could not be generated for the 14th Legislature due to extreme absenteeism and a low volume of ballot. While filtering out deputies with less than 20% participation was necessary to avoid the 'Arch Effect' and data distortion, it resulted in too few data points to provide more than basic legislative insights.


### 4.3 Thematic Analysis

We apply a **thematic classification** based on the title of each ballot vote. Themes include:
- Ecology & Territories (agriculture, climate, energy, transport)
- Economy & State (taxation, customs, inflation)
- Sovereignty & International Affairs (police, justice, defense)
- Solidarity & Social Policy (pensions, social benefits, disability)

This enables **theme-based analyses**.

#### Categorization Methodology

We implemented a **deterministic keyword-matching algorithm**. This process filters the legislative titles provided by the NosDéputés.fr XML API to categorize each vote into one of four strategic domains.

> **Methodological Note:** > While a Large Language Model (LLM) would undoubtedly be more "sophisticated" at interpreting the nuanced context of legislative titles, we decided to stick to a keyword-based approach. It is simple and easily understandable.

The script scans each `titre` (title) tag within the XML response. If a keyword is found, the `scrutin_id` (ballot ID) is mapped to that specific theme using the following logic:

```python
THEMATIQUES = {
    "Solidarité & Social": [
        "pauvreté", "handicap", "retraite", "social", "précarité", "apl", 
        "famille", "prestations", "rsa", "solidarité", "chômage"
    ],
    "Écologie & Territoires": [
        "écologie", "environnement", "climat", "nucléaire", "énergie", "biodiversité", 
        "eau", "agriculture", "agricole", "pesticide", "rural", "transport"
    ],
    "Économie & État": [
        "économie", "fiscal", "impôt", "inflation", 
        "douanes", "entreprises", "croissance"
    ],
    "Souveraineté & International": [
        "justice", "sécurité", "police", "prison", "immigration", 
        "étranger", "asile", "frontière", "armée", "défense", "europe"
    ]
}
```

#### Dataset Overview: Distribution of Ballots by Theme

The following table summarizes the volume of ballot votes analyzed for each legislature, categorized by their primary thematic focus. These themes serve as the basis for our comparative spatial analysis.

| Theme | 14th Legislature (2012-2017) | 15th Legislature (2017-2022) | 16th Legislature (2022-2024) |
| :--- | :---: | :---: | :---: |
| **Solidarity & Social** | 138 | 757 | 335 |
| **Ecology & Territories** | 73 | 641 | 709 |
| **Economy & State** | 72 | 157 | 68 |
| **Sovereignty & International** | 26 | 396 | 519 |
| **Total Analyzed Ballots** | **309** | **1,951** | **1,631** |

There has been a notable rise in MP activity in recent years: the number of ballots grew from 1,023 during the 14th Legislature to 4,394 in the 15th and 4,029 in the 16th.

#### PCA by Theme

Beyond the global PCA, we repeat the analysis for each thematic domain. For example, for *“Solidarity & Social”*:

1. Filter $\mathbf{M}$ to retain only ballot votes labeled “solidarity” or "social"
2. Reapply PCA to this submatrix
3. Visualize: Points colored by political group
4. Observe: **Theme-specific cleavages**

Not all themes yield insightful visualizations; those with higher explained variance in the PCA are the most significant for analysis.

| 15th Legislature (2017-2022) | 16th Legislature (2022-2024) |
| :---: | :---: |
| <a href="L15_pca_Solidarité_&_Social.png" target="_blank"><img src="L15_pca_Solidarité_&_Social.png" style="width: 100%; max-height: 500px; object-fit: contain; cursor: zoom-in;" alt="PCA L15 Social"></a> | <a href="L16_pca_Solidarité_&_Social.png" target="_blank"><img src="L16_pca_Solidarité_&_Social.png" style="width: 100%; max-height: 500px; object-fit: contain; cursor: zoom-in;" alt="PCA L16 Social"></a> |

**Figure 6:** Principal Component Analysis for ballot votes related to Solidarity and Social.

**Main observations:**

- 16th Legislature - The LFI vs. LR Cleavage: The PCA identifies a clear opposition between LFI and LR. This represents the classic "Social vs. Liberal" divide regarding welfare, labor laws, and state intervention.

- Regarding social issues, the RN exhibits a significant shift toward LFI's positions, creating a shared oppositional front against the LREM-LR nexus. This alignment highlights a clear divide between interventionist model and the liberal framework favored by the presidential majority and the traditional right.

- The explained variance of the first principal component (PC1) serves as a proxy for how "structured" or predictable a political cleavage is:
    - 15th Legislature (PC1 ≈ 20%): Social issues were relatively fluid, with more heterogeneous voting patterns across the chamber.
    - 16th Legislature (PC1 > 40%): The doubling of the explained variance indicates a massive increase in political polarization.

---

While PCA highlights political blocs, a fragmented assembly often requires transversal compromises to reach a majority. To identify the specific actors who facilitate these compromises, we use **Betweenness Centrality**. This metric moves beyond simple group membership to pinpoint deputies who act as mandatory "bridges" between different ideological clusters.

### 5 Identification of Strategic Pivots (Betweenness Centrality)

Unlike simple popularity (Degree), this metric identifies deputies who act as mandatory "bridges" between different ideological clusters.

#### 5.1. Mathematical Definition
The centrality $g(v)$ of an MP $v$ is calculated by counting how many shortest paths between all other pairs of MPs pass through $v$:

$$g(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Where:
* $\sigma_{st}$ is the total number of shortest paths from MP $s$ to MP $t$.
* $\sigma_{st}(v)$ is the number of those paths that pass through $v$.

#### 5.2. Distance Inversion and Pathfinding
Since our graph edges represent **similarity** (Cosine Similarity), we must transform them into **distances** to find the "shortest" ideological path. We define the distance $d_{ij}$ as:

$$d_{ij} = \frac{1}{\text{weight}_{ij} + \epsilon}$$

This inversion ensures that a high voting similarity results in a short distance. The algorithm then identifies "pivots": deputies who, by their transversal voting patterns, minimize the distance between antagonistic groups (e.g., bridging the gap between the Majority and the Opposition).

#### 5.3. Political Significance

In a parliament without an absolute majority, these MPs represent the **connective tissue** of the institution. A high Betweenness score reveals a **brokerage capacity**: these individuals are structurally positioned to negotiate amendments that can "swing" a vote, as they constitute the most probable pathway for a ballot to transition from one political bloc to another.

| Rank | 15th Leg. (2017-2022) | Group | | 16th Leg. (2022-2024) | Group |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **1** | Jean-Luc Warsmann | UDI_I | | Nathalie Bassire | LIOT |
| **2** | Karine Lebon | GDR | | Emmanuelle Ménard | NI |
| **3** | Jennifer De Temmerman | LT | | Jean-Carles Grelier | REN |
| **4** | Thierry Michels | LREM | | Olivier Serva | LIOT |
| **5** | Agnès Thill | UDI_I | | Jean-Victor Castor | GDR |
| **6** | Lise Magnier | AGIR-E | | Mansour Kamardine | LR |
| **7** | Jean-Philippe Nilor | GDR | | David Habib | NI |
| **8** | Charles de Courson | LT | | Charles de Courson | LIOT |
| **9** | Brigitte Bourguignon | LREM | | Victor Catteau | RN |
| **10** | Paul Christophe | AGIR-E | | Laurent Panifous | LIOT |

In the largest groups (REN, LFI, RN), the high level of party discipline mechanically reduces the betweenness score of individual members.
Conversely, deputies from smaller or more heterogeneous groups (LIOT, NI, UDI) exhibit the highest betweenness scores. Their "intermediate" positioning and lower exposure to rigid party whips allow them to act as strategic variables.

## 6. Conclusion:

### 6.1 Geometric Reinterpretation of French Political Cleavages

Our quantitative findings provide a mathematical framework to classical qualitative political science, while highlighting new structural shifts:

1. **The Collapse of Linear Bipolarity**
   - **Observation:** The geometric structure of the 16th legislature is multi-polar (3 to 4 distinct branches radiating from the origin).
   - **Implication:** The traditional Left-Right spectrum is insufficient to describe modern dynamics. Alliances are now non-linear; for instance, the "ends" of the spectrum (LFI and RN) may occasionally converge on specific social-related votes, creating a "horseshoe" effect that a simple line cannot represent.

2. **The Primacy of the Institutional Cleavage**
   - **Observation:** PC1 consistently reflects the **Executive-Legislative tension** (Majority vs. Opposition)
   - **Insight:** The "logic of the bloc" (supporting or opposing the government) remains the most powerful statistical predictor of voting behavior, often overriding personal or thematic nuances.


### 6.2 The Strategic Function of Pivots

Pivots (identified by high **Betweenness Centrality**) act as the "connective tissue" of the Assembly. They reduce the distance between antagonistic groups. Without enough bridges, the Assembly would more likely reach a state of legislative paralysis.

### 6.3 Limitations: The "Hidden" Dimensions

While the 2D PCA captures the most visible signals (often ~15-20% of total variance), it intentionally discards the "noise" which often contains crucial secondary information. The remaining variance (axes 3 to $n$) typically hides:

- **Cross-cutting cleavages** (e.g., regional vs. national interests)
- **Personal rivalries** (e.g., intra-party factions)
- (...)


### 6.4 Reproducibility

This study demonstrates the **feasibility and utility** of combined quantitative techniques (graphs + PCA) to uncover political structure.
The full source code is available in the GitHub repository: [Networks-Analysis](https://github.com/Ines2r/Networks-Analysis)

---

## References & Data Sources

* **NosDéputés.fr API** [https://www.nosdeputes.fr/api/](https://www.nosdeputes.fr/api/)  
  *Provides access to parliamentary activities and metadata.*

* **Assemblée Nationale Open Data Portal** [https://data.assemblee-nationale.fr/](https://data.assemblee-nationale.fr/)  
  *Official repository for voting records, law proposals, and legislative history.*

---
## Appendices

### Appendix A: Intra-Group Analysis

For each political group $P$, we compute two distinct metrics:

1. **Cohesion Leader (Intra-Group)**:
   - Restrict the graph to members of $P$ only
   - Identify the node with the maximum weighted degree in this subgraph
   - Interpretation: This MP represents the group’s median behavior. They are the most representative of the collective line, voting in perfect synchronization with the majority of their peers.

2. **Hub Leader (Global)**:
   - Identify the node from the group with the maximum weighted degree in the full graph
   - Interpretation: This MP not only aligns with their own group but also shares the highest number of common votes with the rest of the Assembly. They sit at the intersection of the party line and the broader parliamentary average.


<div align="center">

| Group | Cohesion Leader (Intra) | Hub Leader (Global) |
| :--- | :--- | :--- |
| **LREM** | Marie-Christine Verdier-Jouclas | Marie-Christine Verdier-Jouclas |
| **LR** | Bernard Deflesselles | Bernard Deflesselles |
| **SOC** | Christine Pires-Beaune | Christine Pires-Beaune |
| **LFI** | Mathilde Panot | Mathilde Panot |

**Figure 3:** Key Leaders for the 15th Legislature
</div>

<br>

<div align="center">

| Group | Cohesion Leader (Intra) | Hub Leader (Global) |
| :--- | :--- | :--- |
| **REN** | Claire Guichard | Claire Guichard |
| **RN** | Victor Catteau | Victor Catteau |
| **LFI-NUPES** | Anne-Stambach-Terrenoir | Anne-Stambach-Terrenoir |
| **LR** | Jean-Jacques Gaultier | Michel Herbillon |

**Figure 4:** Key Leaders for the 16th Legislature
</div>

The leaders identified in our tables (Figures 3 & 4) are often not household names. While Mathilde Panot is a notable exception, most "hubs" are relatively obscure backbenchers. This discrepancy reveals a potential limitation of our mathematical model.

**The "Participation Bias"**

Our model relies on Cosine Similarity, which heavily weights participation frequency:

- The "Workhorse" Effect: Figures like Claire Guichard or Victor Catteau are almost always in the chamber. By being present for every vote and strictly following instructions, they become the mathematical "center of mass" for their party.

- The "Media Gap": Famous leaders (e.g., Le Pen, Mélenchon) often have lower legislative footprints because they are busy with national media or strategic travel. Our metric may be mistaking constant presence for actual political power.

Methodological Limits: Is Cosine Enough?
We must remain critical of our choice of metric. While Cosine Similarity effectively handles absences, it may also be creating an optical illusion:

Artificial Hubs: Is an MP a "Hub" because they lead others, or simply because they are the most "average" and frequent voter in their group?

In short, our graph may be mapping legislative discipline rather than political influence. While Mathilde Panot proves that one person can be both a media leader and a legislative hub, for most parties, the "real" power likely lies outside the mathematical center of our clusters.

### Appendix B: Architecture and Implementation of Data Retrieval

### 1 Data Source and API

**Primary source:** NosDéputés.fr — a freely accessible collaborative database, fed by the official data of the French National Assembly via its XML export protocols.

**API endpoints:**
```
https://www.nosdeputes.fr/{LEGISLATURE}/scrutins/xml
https://www.nosdeputes.fr/{LEGISLATURE}/scrutin/{SCRUTIN_ID}/xml
```

where `LEGISLATURE` $\in \{15, 16\}$ and `SCRUTIN_ID` is the numerical identifier of the vote.
Unfortunately, the API hasn't the same amount of data for previous legislatures. For the 14th legislature, we found an archive on [Asssemblée Nationale](https://data.assemblee-nationale.fr/).

### 2 Parallel Download Protocol

To accelerate data collection (approx 4,000 ballot votes), we use a **ThreadPoolExecutor** with up to 10 concurrent workers.

**Output:** Three CSV files generated
- `dataset_scrutins_14.csv` (2012–2017)
- `dataset_scrutins_15.csv` (2017–2022)
- `dataset_scrutins_16.csv` (2022–2024)

Each record: `{depute, group, position, scrutin_id}`

### 3 Transformation into a Pivot Matrix

The raw list of votes is transformed into a **sparse matrix**:

```python
pivot_votes = df.pivot_table(
  index='depute', 
  columns='scrutin_id', 
  values='vote_val'
)
```

---
