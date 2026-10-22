# sweettv-recommender-midterm
# SWEET.TV TV Program Recommender — Midterm Quiz

This repository contains the solution for the LLM4Rec Midterm Quiz (SWEET.TV TV program recommender). 

**Final Kaggle Public Score: 0.43507 (Private: 0.43380)**

## Approach
The final solution uses a **Two-Layer recommendation system**:
1. **Candidate Generation:** Up to 200 candidates per user generated from three sources: 
   - Top-50 historical repeats (items the user already watched in train).
   - Top-100 items from Matrix Factorization (ALS).
   - Top-50 globally popular items.
2. **Ranking:** A LightGBM binary classifier (`n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`) trained on 23 features (e.g., user watch count, recency, ALS score, popularity, category, number of airings in the target period).

## Data Pipeline
Viewing logs are joined with the TV schedule using DuckDB (channel ID + time interval overlap). We calculate `screen_time = overlap_seconds / program_duration` and keep only interactions where `screen_time >= 0.8`, matching the ground truth generation process.

## Results (Validation MAP@5)
| Model | MAP@5 (Strict) | Recall@5 |
|---|---|---|
| Popularity Baseline | 0.2779 | 0.3233 |
| ALS (Baseline) | 0.3029 | 0.3424 |
| **Two-Layer (ALS + LGBM)** | **0.3815** | **0.4026** |

## Repository Structure
- `build_interactions.py` — Data engineering: joins logs and schedule, calculates `screen_time`, outputs `interactions.parquet`.
- `recsys_models.py` — Baseline models (Popularity, ALS, CBF) and MAP@5 evaluation.
- `two_layer.py` — The main pipeline: candidate generation, feature engineering, LightGBM training, and final submission generation.
- `submission.csv` — The final submission file that achieved the Public Score of 0.43507.
- `requirements.txt` — Required Python libraries.

## How to Reproduce
1. Place the competition CSV files (`dataset11-30.csv`, `export_arh_*.csv`, `submission_template.csv`) in the same directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
