# SALUS Pipeline Results

This report summarizes the metrics generated from the SALUS pipeline.

## Data Sources

| Category | Name | Path | Rows |
|:---------|:-----|:-----|-----:|
| query | sqld_dev | `data/query/sqld_dev.json` | 1534 |
| query | dev | `data/query/dev.json` | 1534 |
| query | BIRD_1106_dev | `data/query/BIRD_1106_dev.json` | 1534 |
| processed | frontier | `src/data/processed_query/bird_dev_frontier.json.gz` | 1534 |
| processed | frontier_november | `src/data/processed_query/bird_dev_nov_4agent.json.gz` | 1534 |
| processed | frontier_spider | `src/data/processed_query/spider_dev_frontier.json.gz` | 1034 |
| processed | legacy | `src/data/processed_query/bird_dev_legacy.json.gz` | 1534 |
| ground_truth | BIRD_CLEAN_xs | `src/data/ground_truth/BIRD_CLEAN_xs.csv` | 298 |
| ground_truth | GT_RS_200 | `src/data/ground_truth/GT_RS_200.csv` | 200 |
| ground_truth | GT_RS_200_nov_bird | `src/data/ground_truth/GT_RS_200_nov_bird.csv` | 200 |
| ground_truth | GT_RS_200_spider | `src/data/ground_truth/GT_RS_200_spider.csv` | 200 |
| weak_supervision | frontier | `src/data/weak_supervision/frontier` | 1534 |
| weak_supervision | frontier_november | `src/data/weak_supervision/frontier_november` | 1534 |
| weak_supervision | frontier_spider | `src/data/weak_supervision/frontier_spider` | 1034 |
| weak_supervision | legacy | `src/data/weak_supervision/legacy` | 1534 |

## Decision Plane

| Config | Selected | GT Set | F1 | Precision | Recall | TP | FP | FN | TN |
|:-------|:---------|:-------|---:|----------:|-------:|---:|---:|---:|---:|
| frontier | `cart2 + score_fusion` | BIRD_CLEAN_xs | 0.9194 | 0.9112 | 0.9277 | 154 | 15 | 12 | 117 |
| frontier | `cart2 + score_fusion` | GT_RS_200 | 0.8552 | 0.8052 | 0.9118 | 62 | 15 | 6 | 117 |
| legacy | `cart2 + weighted_voting` | BIRD_CLEAN_xs | 0.8412 | 0.8218 | 0.8614 | 143 | 31 | 23 | 101 |
| legacy | `cart2 + weighted_voting` | GT_RS_200 | 0.7389 | 0.6517 | 0.8529 | 58 | 31 | 10 | 101 |
| frontier_november | `cart2 + stacked_meta` | GT_RS_200_nov_bird | 0.8429 | 0.8310 | 0.8551 | 59 | 12 | 10 | 119 |
| frontier_spider | `cart2 + stacked_meta` | GT_RS_200_spider | 0.8750 | 0.8235 | 0.9333 | 56 | 12 | 4 | 128 |

## Frontier Full-Benchmark Detected Error Rates

| Benchmark | Source | Tasks | Predicted Error | Predicted Correct | Detected Error Rate |
|:----------|:-------|------:|----------------:|------------------:|--------------------:|
| sqld_dev | `src/data/processed_query/bird_dev_frontier.json.gz` | 1534 | 656 | 878 | 0.4276 |
| november_bird | `src/data/processed_query/bird_dev_nov_4agent.json.gz` | 1534 | 571 | 963 | 0.3722 |
| spider_dev | `src/data/processed_query/spider_dev_frontier.json.gz` | 1034 | 325 | 709 | 0.3143 |

## Baselines

| GT Set | Family | Method | F1 | Precision | Recall | TP | FP | FN | TN |
|:-------|:-------|:-------|---:|----------:|-------:|---:|---:|---:|---:|
| BIRD_CLEAN_xs | frontier | gpt5 | 0.8614 | 0.8439 | 0.8795 | 146 | 27 | 20 | 105 |
| BIRD_CLEAN_xs | frontier | gpt4o | 0.8249 | 0.8129 | 0.8373 | 139 | 32 | 27 | 100 |
| BIRD_CLEAN_xs | frontier | claude | 0.8616 | 0.9013 | 0.8253 | 137 | 15 | 29 | 117 |
| BIRD_CLEAN_xs | frontier | gemini | 0.8157 | 0.8182 | 0.8133 | 135 | 30 | 31 | 102 |
| BIRD_CLEAN_xs | frontier | majority_vote | 0.8895 | 0.8596 | 0.9217 | 153 | 25 | 13 | 107 |
| BIRD_CLEAN_xs | frontier | majority_vote_3agent | 0.8876 | 0.8721 | 0.9036 | 150 | 22 | 16 | 110 |
| BIRD_CLEAN_xs | legacy | gpt4omini | 0.7229 | 0.7229 | 0.7229 | 120 | 46 | 46 | 86 |
| BIRD_CLEAN_xs | legacy | gpt4o_old | 0.8246 | 0.8011 | 0.8494 | 141 | 35 | 25 | 97 |
| BIRD_CLEAN_xs | legacy | haiku3 | 0.7087 | 0.7066 | 0.7108 | 118 | 49 | 48 | 83 |
| BIRD_CLEAN_xs | legacy | haiku | 0.8012 | 0.8012 | 0.8012 | 133 | 33 | 33 | 99 |
| BIRD_CLEAN_xs | legacy | majority_vote | 0.8436 | 0.7865 | 0.9096 | 151 | 41 | 15 | 91 |
| BIRD_CLEAN_xs | legacy | majority_vote_3agent | 0.8035 | 0.7722 | 0.8373 | 139 | 41 | 27 | 91 |
| BIRD_CLEAN_xs | frontier | stacking_hc | 0.9021 | 0.8889 | 0.9157 | 152 | 19 | 14 | 113 |
| BIRD_CLEAN_xs | legacy | stacking_hc | 0.8605 | 0.8315 | 0.8916 | 148 | 30 | 18 | 102 |
| GT_RS_200 | frontier | gpt5 | 0.7582 | 0.6824 | 0.8529 | 58 | 27 | 10 | 105 |
| GT_RS_200 | frontier | gpt4o | 0.7013 | 0.6279 | 0.7941 | 54 | 32 | 14 | 100 |
| GT_RS_200 | frontier | claude | 0.7612 | 0.7727 | 0.7500 | 51 | 15 | 17 | 117 |
| GT_RS_200 | frontier | gemini | 0.7020 | 0.6386 | 0.7794 | 53 | 30 | 15 | 102 |
| GT_RS_200 | frontier | majority_vote | 0.7843 | 0.7059 | 0.8824 | 60 | 25 | 8 | 107 |
| GT_RS_200 | frontier | majority_vote_3agent | 0.7919 | 0.7284 | 0.8676 | 59 | 22 | 9 | 110 |
| GT_RS_200 | legacy | gpt4omini | 0.6012 | 0.5158 | 0.7206 | 49 | 46 | 19 | 86 |
| GT_RS_200 | legacy | gpt4o_old | 0.6879 | 0.6067 | 0.7941 | 54 | 35 | 14 | 97 |
| GT_RS_200 | legacy | haiku3 | 0.5644 | 0.4842 | 0.6765 | 46 | 49 | 22 | 83 |
| GT_RS_200 | legacy | haiku | 0.6351 | 0.5875 | 0.6912 | 47 | 33 | 21 | 99 |
| GT_RS_200 | legacy | majority_vote | 0.7024 | 0.5900 | 0.8676 | 59 | 41 | 9 | 91 |
| GT_RS_200 | legacy | majority_vote_3agent | 0.6788 | 0.5773 | 0.8235 | 56 | 41 | 12 | 91 |
| GT_RS_200 | frontier | stacking_hc | 0.8243 | 0.7625 | 0.8971 | 61 | 19 | 7 | 113 |
| GT_RS_200 | legacy | stacking_hc | 0.7355 | 0.6552 | 0.8382 | 57 | 30 | 11 | 102 |

## Calibrated benchmark audit

Each estimate uses the manually labeled 200-task sample of the corresponding benchmark version, only after SALUS predictions are fixed.

| Benchmark | Sample F1 | Raw detected rate | Calibrated error rate | 95% CI |
|:--|--:|--:|--:|:--|
| sqld_dev | 0.8552 | 42.76% | 39.34% | [33.21%, 45.47%] |
| november_bird | 0.8429 | 37.22% | 36.76% | [30.22%, 43.30%] |
| spider_dev | 0.8750 | 31.43% | 26.97% | [21.39%, 32.55%] |

