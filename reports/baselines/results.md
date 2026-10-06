# Automatic baseline evaluation

Positive class: annotation error (label 0). Both baselines use automatic outputs only.

| Dataset | Method | F1 | Precision | Recall | TP | FP | FN | TN |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| BIRD_CLEAN_xs | SQLDriller | 0.6456 | 0.6800 | 0.6145 | 102 | 48 | 64 | 84 |
| BIRD_CLEAN_xs | SAR-Agent (o3) | 0.8389 | 0.8466 | 0.8313 | 138 | 25 | 28 | 107 |
| GT_RS_200 | SQLDriller | 0.5316 | 0.4667 | 0.6176 | 42 | 48 | 26 | 84 |
| GT_RS_200 | SAR-Agent (o3) | 0.7260 | 0.6795 | 0.7794 | 53 | 25 | 15 | 107 |
