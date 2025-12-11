

# Final Report — Predicting Resolution Time for Boston 311 Requests

Video Presentation: https://youtu.be/ymDAISBy8io

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

All raw Boston 311 service request data used in this project (2018–2025) can be downloaded publicly at:

**[https://data.boston.gov/dataset/311-service-requests](https://data.boston.gov/dataset/311-service-requests)**

Because full-year CSVs exceed GitHub’s file size limit, they are **not stored in this repository**. The pipeline expects these files to be placed in the `data/` directory following the naming convention:

```
data/
  Service Requests - 2018.csv
  Service Requests - 2019.csv
  ...
  Service Requests - 2025.csv
```

### Weather Data

Hourly weather data for 2018–2025 is already included in the repository under:

```
data/boston_hourly_2018_2025.csv
```

These observations are merged into the pipeline on the hour of request creation to incorporate temperature, precipitation, humidity, wind, and related variables.

### Spatial Data

Spatial context is captured algorithmically from the latitude/longitude values in the 311 dataset, including distance to City Hall.

### Directory Layout

```
CS506-proj/
│
├── data/
│   ├── (Your 311 CSVs here)
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
* **Weather features:** temperature, precipitation, snow, wind, humidity.
* **Text features:** TF-IDF of the request subject (disabled for the main run).

### Exploratory Visualizations

The pipeline automatically generates:

* Resolution time distribution
* Box plots by top request reasons
* Monthly median trends from 2018–2025

These summaries confirm skew, seasonal patterns, and category-level variation.

---

## 5. Modeling Approach

### Temporal Split (to avoid leakage)

* **Train:** 2018–2022
* **Validation:** 2023
* **Test:** 2024–2025

### Target Variable

The model predicts on the transformed target:

```
log1p(winsorized_duration_hours)
```

Predictions are converted back using `expm1` and clipped at the train 99th percentile.

### Models Trained

* **RidgeCV** – linear baseline
* **LassoCV** – sparse linear model
* **HistGradientBoostingRegressor (HGBR)** – final nonlinear model

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

The HGBR model performs the best overall, delivering strong accuracy and capturing nonlinear patterns across time, weather, spatial variation, and categories.

---

## 7. Interpretability and Diagnostics

### Permutation Feature Importance

Ranks signal strength across:

* request category
* workload volume
* weather variables
* day and hour effects
* neighborhood
* distance from City Hall

### Partial Dependence Plots

Show how predictions change with:

* open hour
* day of week
* request volume
* temperature and precipitation
* holiday effects
* spatial distance

### Residual Diagnostics

Residuals reveal:

* good central accuracy (median error ~7 hours)
* error concentration in long-duration categories
* nonlinear structure well-captured by HGBR

---

## 8. Summary of Findings

The analysis shows that Boston 311 resolution times are heavily right-skewed, with most requests closing within a reasonable window but a small set of long-duration cases dominating overall error metrics. Category-level differences explain much of this variation: requests involving Trees, Street Lights, and infrastructure repairs consistently take longer due to inspections, scheduling constraints, or coordination across departments. Temporal patterns—such as late-night openings, weekends, and holidays—also contribute meaningfully to delays, reflecting staffing schedules and reduced operational capacity during certain periods. Weather adds another layer of variability: precipitation, extreme temperatures, and adverse conditions tend to slow down field-dependent activities.

Spatial and operational factors further shape resolution times. Neighborhood differences and distance from central areas introduce logistical complexity, while the 7-day rolling request volume provides a strong indicator of workload pressure, with higher volumes correlating with slower closures. Nonlinear modeling (HGBR) captures these interactions more effectively than linear baselines, producing lower median errors and more stable performance across temporal splits. Although the overall MAE remains above the 24-hour target because of persistent long-tail outliers, the median error of around seven hours demonstrates that the model predicts the majority of requests accurately, and that remaining error is concentrated in specific operationally complex categories rather than due to random model failures.


