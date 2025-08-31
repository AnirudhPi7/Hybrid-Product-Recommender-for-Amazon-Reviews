# Hybrid Product Recommender for Amazon Reviews

---

## Introduction
This project builds and evaluates a personalized product recommender on a cleaned Amazon reviews dataset, comparing a custom hybrid (collaborative + content) system to a LightFM model that unifies interactions and side information. We optimize for top-N relevance using Precision@10, Recall@10, and F1@10, reflecting the business goal of showing the “right ten” items rather than predicting raw ratings. The study positions the hybrid as a transparent baseline and LightFM as a rank-optimizing, feature-aware alternative on the same data and splits. In doing so, it delivers a pragmatic blueprint for moving from explainable heuristics to a production-viable learned ranker.

---

## Background
Collaborative filtering (CF) discovers “people who liked X also liked Y” patterns in the user–item matrix, while content-based filtering (CBF) recommends by item similarity derived from metadata—useful for cold-start but blind to community taste. A hybrid balances these strengths, blending CF’s latent preference structure with CBF’s explainable, metadata-driven links; LightFM goes further by learning both signals together and directly optimizing top-K ranking. This framing also clarifies interpretability: CF factors are abstract, CBF can be traced to concrete words/fields, and LightFM sits in between while handling new items through features.

---

## Problem Statement
We set out to design a top-10 recommender that remains robust with sparse interactions and evolving catalogs, and to test whether a unified ranker (LightFM) can outperform a transparent hybrid baseline on identical data. Concretely, we compare a weighted CF+CBF scorer against LightFM trained on the binarized interaction matrix with item features, judging success by improvements in Precision@10/Recall@10/F1@10 and by practical cold-start behavior. The outcome should guide an implementation path that starts simple and explainable, then graduates to rank-optimized, feature-aware modeling where justified.

---

## Dataset
The corpus is a cleaned subset of Amazon reviews with user_id, asin, explicit ratings, and rich product metadata (title, main_category, store, price, features, details, description). After preprocessing and sparsity controls, the working matrix spans ~66k users, ~5k products, and 68k+ ratings; we retain users and items with ≥ 3 interactions to stabilize training and evaluation. For ranking models that require implicit feedback, ratings are binarized with ≥ 2.5 treated as positive interactions. These choices reduce extreme sparsity, align targets with top-N ranking, and preserve enough coverage for side-feature modeling.

---

## Data Preparation
We assemble a user–item interaction matrix for CF and a consolidated text field for CBF from product title, category, description, price, store, features, and details, then vectorize text with TF-IDF and apply LSA (Truncated SVD) for dense item embeddings. Typical dimensionalities are compact (e.g., k ≈ 20 CF factors; ≈100-dim LSA vectors) to keep scoring fast while retaining signal. These representations support scalable dot-product scoring on the CF side and cosine similarity on the CBF side, with all features kept sparse/dense as appropriate for efficient computation.

---

## Methodology
The custom hybrid learns CF factors via Truncated SVD on the interaction matrix and computes predicted scores; in parallel, CBF builds TF-IDF → LSA item vectors and measures cosine similarity to derive a content score. We normalize both signals and blend them with a fixed weight — 60% CF + 40% CBF — then rank items per user; for true cold-start, a popularity fallback surfaces widely rated products until interactions accrue. In contrast, LightFM trains a single model on binarized interactions with item features (category, store or TF-IDF/LSA), using WARP loss to target top-K ranking directly, with components and learning rate tuned across {32,64,128} and {0.01, 0.05, 0.10}.

---

## Evaluation
The hybrid achieved P@10 = 0.1323, R@10 = 0.3003, F1@10 = 0.1823, a balanced yet moderate ranking quality consistent with its linear blend and compact factors. LightFM with WARP, 128 components, and learning_rate = 0.10 substantially improved coverage with P@10 ≈ 0.20 and R@10 ≈ 0.60, leveraging side features and pairwise ranking to elevate relevant items into the top-10. In comparative analysis, LightFM’s gains stem from jointly learning how content features shape collaborative signals, whereas the hybrid’s linear fusion and smaller latent dimension limit expressiveness; nonetheless, the hybrid remains transparent and performant as a baseline.

---

## Assumptions & Practical Considerations
Two pragmatic assumptions underlie the pipeline: binarization of explicit ratings into implicit “likes,” and sparsity filtering of low-activity users/items; both stabilize learning but shift the target from cardinal scores to top-N relevance. The hybrid assumes a linear trade-off between CF and CBF; LightFM relaxes this by learning feature-aware embeddings while explicitly optimizing rank, which mitigates some linear-blend limitations. Finally, cold-start is handled by metadata-driven CBF and a popularity fallback, ensuring reasonable recs even before CF has signal; this design decision prioritizes user experience during ramp-up and aligns with production constraints.

---
## Conclusion & Future Scope
A clear path emerges: start with the hybrid for transparency and quick wins, then promote to LightFM (WARP) when side features are reliable and top-K uplift justifies complexity. Future extensions include tuning hybrid non-linearly, enriching item features with learned text embeddings, and expanding LightFM searches over components/loss/lr to chase marginal gains while monitoring online relevance. This staged approach balances explainability, cold-start robustness, and ranking performance for a real-world recommender.

---
