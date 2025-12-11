# Final Report — Predicting Resolution Time for Boston 311 Requests

Video Presentation: **

---

## 1. Introduction

This project predicts how long a Boston 311 service request will take to close, measured in hours from the initial creation timestamp. The motivation is twofold: to help residents set realistic expectations and to help the city identify potential operational bottlenecks. The primary objective is to forecast resolution time with a mean absolute error (MAE) of ≤24 hours on a strictly held-out temporal test set. This final report summarizes the complete end-to-end pipeline, including data collection, processing, modeling, evaluation, interpretability, and reproducibility setup.

---

## 2. How to Build and Run the Code

All commands assume the project root directory.

### Installation

```bash
make setup
```

This creates a virtual environment and installs all dependencies listed in `requirements.txt`.

### Running the Full Pipeline

```bash
make run
```

This executes `midterm_pipeline.py` end-to-end, generating:

* All preprocessing steps
* Weather / holiday / spatial merging
* EDA visualizations
* All three models (RidgeCV, LassoCV, HGBR)
* Predictions for validation and test sets
* Feature importance, partial dependence plots, and residual diagnostics
* `experiment_results.csv`

### Running Tests

```bash
make test
```

A lightweight set of automated tests validates feature engineering, weather and holiday merging, and the ability of the model pipeline to generate predictions.

The GitHub Actions workflow executes the same tests automatically on every push.

---

## 3. Data Sources and Structure

### 311 Service Request Data

Raw files (CSV) for **2018–2025** were downloaded from the City of Boston’s open data portal. Relevant fields include timestamps, category/reason/type, location coordinates, neighborhood, and source channel.

### Weather Data

Hourly weather observations for 2018–2025 were collected and merged on the hour of request creation. Weather variables included temperature, precipitation, humidity, wind, and related atmospheric attributes.

### Spatial Data

A simple proxy for spatial context was computed as the distance between the request coordinates and Boston City Hall, expressed in kilometers.

### Directory Layout

```
CS506-proj/
│
├── data/
│   ├── 311_2018.csv
│   ├── ...
│   └── boston_hourly_2018_2025.csv
│
├── report/
│   ├── experiment_results.csv
│   └── figures/
│       ├── 01_duration_hist.png
│       ├── 02_box_by_reason.png
│       ├── 03_monthly_median_trend.png
│       ├── importance_permutation.png
│       ├── partial_dependence_*.png
│       └── residual_plot.png
│
├── midterm_pipeline.py
├── requirements.txt
└── Makefile
```

---

## 4. Data Processing

### Cleaning Steps

* Standardized column names and parsed all temporal fields.
* Removed invalid, missing, or negative durations.
* Deduplicated entries by `case_id`.
* Winsorized resolution times at the 99th percentile to reduce heavy-tail effects.
* Validated geographic coordinates and set invalid ones to missing.

### Feature Engineering

* **Temporal variables:** hour of day, day of week, month, and weekend flag.
* **Workload proxy:** 7-day rolling average of recent request volume.
* **Holiday flag:** derived using the `holidays` Python package.
* **Spatial proxy:** approximate distance to City Hall in kilometers.
* **Weather features:** temperature, precipitation, snow, wind, humidity (merged by hour).
* **Text features:** TF-IDF of the request subject (disabled for the main run due to scale).

### Exploratory Visualizations

The pipeline automatically generates:

* Distribution of resolution times (clipped).
* Box plots for the top request reasons.
* Monthly trend in median resolution time from 2018–2025.

These plots help confirm skew, seasonal behavior, and category-level variation.

---

## 5. Modeling Approach

### Temporal Split (to avoid leakage)

* **Train:** 2018–2022
* **Validation:** 2023
* **Test:** 2024–2025

### Target Variable

The model predicts a transformed target:

```
log1p(winsorized_duration_hours)
```

Predictions are inverted with `expm1` and clipped at the train 99th percentile.

### Models Trained

* **RidgeCV** – linear baseline
* **LassoCV** – sparse linear baseline
* **HistGradientBoostingRegressor (HGBR)** – nonlinear tree-based final model

Categorical variables were one-hot encoded, numeric variables were passed through a median imputer, and optional text features were handled via TF-IDF.

---

## 6. Final Results

| Model   | Split          | MAE (hrs) | RMSE (hrs) | Median AE (hrs) |
| ------- | -------------- | --------- | ---------- | --------------- |
| RidgeCV | Val 2023       | 7505.30   | 10910.08   | 1127.74         |
|         | Test 2024–2025 | 10059.44  | 12785.42   | 16443.95        |
| LassoCV | Val 2023       | 316.11    | 1218.54    | 12.09           |
|         | Test 2024–2025 | 133.23    | 488.46     | 7.64            |
| HGBR    | Val 2023       | 305.23    | 1192.61    | 11.61           |
|         | Test 2024–2025 | 127.95    | 467.02     | 7.19            |

The **HGBR** model performs the best overall, providing both strong accuracy and interpretability. Although the MAE remains above the original 24-hour target, the median absolute error is consistently around **7 hours**, showing strong performance for the majority of requests. Long-tail outliers dominate the MAE.

---

## 7. Interpretability and Diagnostics

The interpretability module generates:

### Permutation Feature Importance

Reveals the most influential predictors, including:

* request category
* rolling request volume
* weather variables
* day of week and hour of day
* neighborhood
* spatial distance

### Partial Dependence Plots

Show how predicted duration changes with:

* time of day
* request volume
* temperature and precipitation
* holidays
* spatial distance

### Residual Diagnostics

Residual plots highlight:

* nonlinear patterns captured by HGBR
* higher errors in certain categories (e.g., Trees, Street Lights)
* heteroskedasticity driven by long-duration requests

All interpretability outputs are saved in `report/figures/`.

---

## 8. Summary of Findings

The analysis shows that Boston 311 resolution times are heavily right-skewed, with most requests closing within a reasonable window but a small set of long-duration cases dominating overall error metrics. Category-level differences explain much of this variation: requests involving Trees, Street Lights, and infrastructure repairs consistently take longer due to inspections, scheduling constraints, or coordination across departments. Temporal patterns—such as late-night openings, weekends, and holidays—also contribute meaningfully to delays, reflecting staffing schedules and reduced operational capacity during certain periods. Weather adds another layer of variability: precipitation, extreme temperatures, and adverse conditions tend to slow down field-dependent activities.

Spatial and operational factors further shape resolution times. Neighborhood differences and distance from central areas introduce logistical complexity, while the 7-day rolling request volume provides a strong indicator of workload pressure, with higher volumes correlating with slower closures. Nonlinear modeling (HGBR) captures these interactions more effectively than linear baselines, producing lower median errors and more stable performance across temporal splits. Although the overall MAE remains above the 24-hour target because of persistent long-tail outliers, the median error of around seven hours demonstrates that the model predicts the majority of requests accurately, and that remaining error is concentrated in specific operationally complex categories rather than due to random model failures.

---
