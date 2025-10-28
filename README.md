## Midterm Progress Report — Predicting Resolution Time for Boston 311 Requests

Video Presentation: https://youtu.be/SFe9qG2eDk8

### 1. Introduction

This project aims to predict how long a Boston 311 service request will take to close, measured in hours from its creation timestamp. The broader goal is to provide actionable insights that can help both residents (by setting more realistic expectations) and the city’s operations teams (by identifying potential service bottlenecks). The final objective is to achieve a mean absolute error (MAE) of ≤24 hours on a held-out temporal test set. This midterm report documents the current state of the pipeline, including preprocessing, initial modeling, exploratory data analysis, and baseline results.

---

### 2. Data Processing Done So Far

* Consolidated 311 request CSVs for **2018–2025** and standardized schema by lowercasing column names.
* Parsed timestamps (`open_dt` / `closed_dt`) and computed the target as **hours to closure**.
* Removed rows with missing or illogical durations and deduplicated entries by `case_id`.
* Winsorized `duration_hours` at the 99th percentile to stabilize training under heavy tails.
* Engineered temporal context features: **hour of day**, **day of week**, **month**, and **weekend flag**.
* Added a lightweight operational workload proxy via **7-day rolling average** of recent daily request volume.
* Exported preliminary exploratory plots (distribution, category-level variation, and monthly trends).

---

### 3. Data Modeling Done So Far

* Applied a **chronological split** to avoid temporal leakage:

  * Train: **2018–2022**
  * Validation: **2023**
  * Test (held-out): **2024–2025**
* Trained a fast baseline using a **Ridge regression model** on a **log-transformed** winsorized duration target.
* Categorical variables are one-hot encoded; numeric features are passed through unchanged.
* Text features (TF-IDF from `subject`) are wired but **not yet enabled** to keep the baseline lightweight for ~2.1M rows.
* Predictions are clipped at the 99th percentile of train durations to avoid rare outlier explosions and keep the model stable.

---

### 4. Interpretation of Preliminary Visualizations

* Resolution times are **extremely right-skewed**, confirming the need for both winsorization and log-transform modeling.
* Certain request types, such as **Street Lights** and **Trees**, tend to stay open significantly longer than others, illustrating strong category-level signal.
* The **monthly median resolution time has trended downward** in recent years, especially post-2023, suggesting operational improvements or shifts in request mix.
* These visual patterns justify the chosen baseline features and motivate incorporation of richer context going forward.

---

### 5. Preliminary Results

Using the current baseline Ridge model:

| Split           | MAE (hrs) | RMSE (hrs) | Median AE (hrs) |
| --------------- | --------- | ---------- | --------------- |
| Validation 2023 | 316.97    | 1215.62    | 11.58           |
| Test 2024–2025  | 134.67    | 493.79     | 7.46            |

The baseline performs noticeably better on the held-out **2024–2025** window than on the 2023 validation year, with a **very low median error** indicating that the majority of predictions are tightly clustered around the true value. The remaining error mass stems from long-duration outliers, confirming the need for additional contextual signals such as neighborhood-level spatial features, holidays, weather, and text.

---
