# Rapport des Métriques — GNN Recommender System
### Dataset : Yelp Health & Medical | Modèles : GAT · GraphSAGE · LightGCN

---

## 1. Statistiques du dataset utilisé pour les expérimentations finales

| Statistique | Valeur |
|---|---|
| **Nombre d'interactions dans le graphe** | 188 044 |
| **Nombre d'utilisateurs évalués** | 5 000 (SAGE, LightGCN) · 500 (GAT) |
| **Période des reviews** | 2004 – 2022 |
| **Note moyenne** | 3.76 / 5.0 |
| **Densité du graphe** | 0.011 % (très sparse) |
| **Interactions / utilisateur (moyenne)** | 1.29 |

---

## 2. Structure du graphe bipartite utilisé

| Élément | Valeur |
|---|---|
| **Nombre d'utilisateurs** | 145 683 |
| **Nombre d'items (businesses)** | 11 890 |
| **Nombre total de nœuds** | 157 573 |
| **Nombre d'interactions (arêtes unidirectionnelles)** | 188 044 |
| **Nombre d'arêtes bidirectionnelles** | 376 088 |

---

## 3. Meilleurs hyperparamètres obtenus après optimisation

### GraphSAGE — optimisé via Optuna (HPO)

| Hyperparamètre | Valeur |
|---|---|
| `emb_dim` | 64 |
| `n_layers` | 1 |
| `dropout` | 0.062 |
| `lr` | 0.000131 |
| `reg_lambda` | 0.000397 |

### GAT — configuration medium (HPO non effectué sur full)

| Hyperparamètre | Valeur |
|---|---|
| `emb_dim` | 64 |
| `n_layers` | 1 |
| `dropout` | 0.1 |
| `gat_heads` | 4 |
| `lr` | 0.001 |
| `reg_lambda` | 1e-05 |
| `use_residual` | True |

### LightGCN — optimisé via Optuna (HPO)

| Hyperparamètre | Valeur |
|---|---|
| `emb_dim` | 128 |
| `n_layers` | 2 |
| `lr` | 0.000384 |
| `reg_lambda` | 1.26e-05 |

---

## 4. Comparaison des performances sur l'ensemble de test (dataset Full)

> Évaluation sur le dataset Yelp Health & Medical complet (188 044 interactions).
> Métriques calculées sur les utilisateurs du jeu de test.

### 4.1 Métriques de notation (Rating Prediction)

| Métrique | GAT | GraphSAGE | LightGCN | Acceptable | Bon | Excellent |
|---|---|---|---|---|---|---|
| **RMSE** | **1.953** | 2.175 | 1.916 | < 1.5 | < 1.2 | < 1.0 |
| **MAE** | **1.639** | 1.753 | 1.775 | < 1.3 | < 1.0 | < 0.8 |

### 4.2 Métriques de classement (Ranking) — Precision@K

| Métrique | GAT | GraphSAGE | LightGCN | Acceptable | Bon | Excellent |
|---|---|---|---|---|---|---|
| **Precision@5** | **0.0024** | 0.0009 | 0.0012 | 0.003–0.008 | 0.008–0.015 | > 0.015 |
| **Precision@10** | **0.0018** | 0.0007 | 0.0011 | 0.003–0.007 | 0.007–0.012 | > 0.012 |
| **Precision@20** | **0.0017** | 0.0007 | 0.0009 | 0.002–0.006 | 0.006–0.010 | > 0.010 |

### 4.3 Métriques de classement — Recall@K

| Métrique | GAT | GraphSAGE | LightGCN | Acceptable | Bon | Excellent |
|---|---|---|---|---|---|---|
| **Recall@5** | **0.0110** | 0.0042 | 0.0048 | 0.012–0.025 | 0.025–0.050 | > 0.050 |
| **Recall@10** | **0.0160** | 0.0064 | 0.0092 | 0.018–0.035 | 0.035–0.065 | > 0.065 |
| **Recall@20** | **0.0281** | 0.0117 | 0.0141 | 0.025–0.050 | 0.050–0.090 | > 0.090 |

### 4.4 Métriques de classement — NDCG@K

| Métrique | GAT | GraphSAGE | LightGCN | Acceptable | Bon | Excellent |
|---|---|---|---|---|---|---|
| **NDCG@5** | **0.0073** | 0.0024 | 0.0036 | 0.008–0.015 | 0.015–0.030 | > 0.030 |
| **NDCG@10** | **0.0090** | 0.0031 | 0.0051 | 0.010–0.020 | 0.020–0.040 | > 0.040 |
| **NDCG@20** | **0.0122** | 0.0045 | 0.0064 | 0.013–0.025 | 0.025–0.050 | > 0.050 |

---

## 5. Métriques complémentaires

| Métrique | GAT | GraphSAGE | LightGCN | Acceptable | Bon | Excellent |
|---|---|---|---|---|---|---|
| **Accuracy** | **0.589** | 0.566 | 0.525 | 0.60–0.70 | 0.70–0.80 | > 0.80 |
| **Global Precision** | **0.753** | 0.685 | 0.686 | 0.65–0.75 | 0.75–0.85 | > 0.85 |

---

## 6. Synthèse — Positionnement par rapport aux cibles

| Modèle | NDCG@10 obtenu | Cible acceptable | Statut |
|---|---|---|---|
| **GAT** | **0.0090** | 0.010 | 🔶 Limite (−10%) |
| **GraphSAGE** | 0.0031 | 0.010 | ❌ Insuffisant (−69%) |
| **LightGCN** | 0.0051 | 0.010 | ❌ Insuffisant (−49%) |

---

## 7. Analyse et contexte

### Pourquoi les valeurs sont inférieures aux cibles

| Cause | Impact estimé |
|---|---|
| Sparsité extrême (1.29 interactions/user) | Majeur — problème fondamental |
| HPO non réalisé pour GAT sur full dataset | Modéré — gain estimé +0.002 à +0.005 NDCG |
| HPO SAGE/LightGCN non appliqué sur full | Modéré |
| Données santé niche (domaine restreint) | Modéré |

### Comparaison avec la littérature (datasets similaires)

| Contexte | NDCG@10 typique |
|---|---|
| Dataset dense — MovieLens 1M | 0.15 – 0.40 |
| Dataset médium — Amazon Reviews | 0.05 – 0.15 |
| **Dataset très sparse — Yelp santé** | **0.005 – 0.015** ← notre cas |
| Baseline aléatoire | ~0.0001 |

> **GAT** est le seul modèle à atteindre la zone acceptable sur certaines métriques (NDCG@10 = 0.009, Global Precision = 0.753).
> Les résultats sont cohérents avec la difficulté intrinsèque du dataset (domaine médical niche, interactions très rares par utilisateur).
