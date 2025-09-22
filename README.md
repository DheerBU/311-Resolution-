# Project Proposal — Predicting Resolution Time for Boston 311 Requests

## Description of the project

I will build an end-to-end pipeline that predicts how long a Boston 311 service request will take to close, measured in hours from creation to closure. My workflow will cover data collection, cleaning, feature extraction, visualization, and model training, with the final outcome being accurate, interpretable predictions and supporting analyses.

## Goal

My primary goal is to train a regression model that predicts time-to-close at request creation with a mean absolute error of 24 hours or less on a held-out temporal test set. A complementary goal is to explain the main drivers of resolution time so that the results are interpretable and useful to stakeholders.

## Data Collection

I will collect Boston 311 service request records (open/close timestamps, case ID, category/reason/type, short text subject/description, source channel, latitude/longitude, neighborhood) from the City of Boston Open Data portal via the Socrata API, pulling data in monthly batches and saving raw snapshots (CSV/Parquet). I will gather weather observations (hourly/daily temperature, precipitation, snow, wind) for the same periods from a programmatic source such as Meteostat or NOAA and cache them locally. I will add spatial context by downloading Boston neighborhood polygons (GeoJSON/Shapefile) and performing point-in-polygon joins. I will derive calendar features (holiday flags, weekday/weekend, month/season) directly from timestamps using a Python holidays utility.

## Data modelling

I will compute the target as hours between open and close. I will clean the data by removing or flagging missing/invalid timestamps, deduplicating by case ID, winsorizing extreme durations to reduce heavy-tail effects, standardizing categorical levels, and validating coordinates within Boston bounds. For features, I will engineer temporal indicators (hour, weekday, month/season, holiday, recent rolling request volume), categorical encodings (reason, type, source, neighborhood), weather variables (temperature, precipitation indicators and amounts, snow, wind, prior-24-hour precipitation), spatial proxies (e.g., distance to City Hall), and lightweight text features from the subject field using TF-IDF unigrams/bigrams with a capped vocabulary. I will establish baselines with regularized linear models (Ridge/Lasso), then train gradient-boosted decision trees (e.g., XGBoost/LightGBM) on a log-transformed target to handle skew. I will use feature importance, partial dependence, and SHAP values to interpret both global and local behavior.

## Visualizing the data

I will create exploratory time series showing how median and distributional resolution times evolve by month and by major request categories. I will compare neighborhoods and source channels with box/violin plots and visualize spatial patterns using neighborhood-level choropleths of median and 90th-percentile resolution times, along with point-density heatmaps for request volume. For the models, I will present feature importance rankings, SHAP beeswarm plots, and partial-dependence curves for top predictors. I will also build a simple interactive interface (Streamlit with Plotly/Folium) so I can filter by date range, category, neighborhood, and source, and run “what-if” predictions by editing feature inputs.

## Test plan

To prevent temporal leakage, I will split the data chronologically: train on earlier years, validate on a contiguous subsequent window for model selection and tuning, and reserve a final, later window as the held-out test set. I will evaluate primarily with mean absolute error in hours and also report RMSE and median absolute error to reflect robustness under heavy tails. I will assess performance on the held-out temporal test set and perform stratified evaluations by neighborhood, category, and season. Finally, I will run ablations by removing text, weather, and spatial feature groups to quantify their incremental value on the same temporal splits.
