# Bias-Aware News Recommendation System
 
A bias-aware news recommendation system built on the MIND dataset that integrates a GRU temporal module into the DKN (Deep Knowledge-aware Network) framework, with Jensen-Shannon divergence used to quantify and surface recommendation diversity gaps.
 
## Why this matters
 
News recommendation algorithms are known to amplify filter bubbles, systematically over-recommending content that confirms existing preferences rather than exposing users to diverse viewpoints. This project goes beyond improving recommendation accuracy: it quantifies the *bias* in recommendations using information-theoretic measures, identifying where and how a system diverges from user intent. That kind of measurement infrastructure is a prerequisite for making recommendation systems fairer at scale.
 
---
 
## Approach
 
| Component | Detail |
|---|---|
| Dataset | Microsoft News Dataset (MIND) - real user impression logs + article metadata |
| Base model | DKN (Deep Knowledge-aware Network) with entity-level knowledge graph embeddings |
| Extension | GRU recurrent module for temporal dynamics in user browsing history |
| Bias metric | Jensen-Shannon divergence between user preference and recommendation distributions |
 
---
 
## Key contributions
 
### 1. Temporal modeling via GRU
Integrated a Gated Recurrent Unit (GRU) module into DKN's user encoder to capture the *sequence* and *recency* of article interactions, not just the aggregate. User interests shift over time; a static bag-of-clicks representation misses this. The GRU extension improved personalized recommendation quality over the static DKN baseline.
 
### 2. Bias quantification via Jensen-Shannon divergence
Applied JS divergence to measure the alignment between what users prefer (derived from impression logs) and what the system recommends. Lower divergence = better alignment. The analysis surfaced:
- Categories that were systematically over-recommended relative to user interest
- Users with narrow browsing histories who received disproportionately homogeneous feeds
- Concrete opportunities to increase recommendation diversity without sacrificing relevance
### 3. Fairness analysis
Identified that users with short or narrow browsing histories, a common cold-start variant, received the least diverse recommendations, suggesting that the model was defaulting to popularity-based fallbacks that compounded rather than corrected the bias.
 
---
 
## Results
 
The GRU-augmented model outperformed the static DKN baseline on standard recommendation metrics. JS divergence analysis revealed measurable gaps in recommendation diversity, particularly for narrow-history users, providing a quantitative basis for fairness-aware reranking strategies.
 
---
 
## How to run
 
```bash
pip install -r requirements.txt
 
# Train
python train.py --model dkn_gru --dataset mind
 
# Evaluate with bias metrics
python evaluate.py --metric js_divergence
```
 
---
 
## Repository structure
 
```
news-recommendation-system/
├── train.py              # Training loop - DKN + GRU extension
├── evaluate.py           # Evaluation - accuracy + JS divergence
├── model/
│   ├── dkn.py            # Base DKN architecture
│   └── gru_encoder.py    # GRU temporal user encoder
├── data/
│   └── mind_loader.py    # MIND dataset preprocessing
├── requirements.txt
└── README.md
```
 
---
 
## Skills demonstrated
 
- Deep learning for recommender systems (DKN, GRU, knowledge graphs)
- Bias quantification using information-theoretic metrics (JS divergence)
- Large-scale dataset analysis (MIND: millions of impression logs)
- Fairness evaluation methodology for ranking systems
- Research leadership: bias quantification methodology was an original contribution to the team project
## Context
 
Completed February-May 2024 at the University of Kentucky as part of graduate Data Mining coursework. Role: Project Manager and lead analyst for the bias quantification methodology. The JS divergence framing and fairness analysis were original contributions beyond the assignment baseline.
