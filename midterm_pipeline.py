# midterm_pipeline.py (final with interpretability via permutation importance)
import os, glob, warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV, LassoCV, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

import holidays

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "/Users/dheerdoshi/Documents/CS506-proj/data"
FIG_DIR  = "/Users/dheerdoshi/Documents/CS506-proj/report/figures"
os.makedirs(FIG_DIR, exist_ok=True)
REPORT_DIR = os.path.dirname(FIG_DIR)
os.makedirs(REPORT_DIR, exist_ok=True)

YEARS_TO_USE = list(range(2018, 2026))
VAL_YEAR     = 2023
TEST_YEARS   = [2024, 2025]

WEATHER_COLS = ["temp", "dwpt", "rhum", "prcp", "wspd", "wdir", "pres"]

USE_TEXT      = False
USE_WEATHER   = True
USE_HOLIDAYS  = True
USE_SPATIAL   = True

FAST_MODE     = False
PRED_CAP_PCT  = 0.99

COL_OPEN   = "open_dt"
COL_CLOSE  = "closed_dt"
COL_ID     = "case_id"
COL_SUBJ   = "subject"
COL_REASON = "reason"
COL_TYPE   = "type"
COL_SRC    = "source"
COL_LAT    = "latitude"
COL_LON    = "longitude"
COL_NHOOD  = "neighborhood"

CITY_HALL_LAT, CITY_HALL_LON = 42.3601, -71.0589
US_HOLIDAYS = holidays.US()

EXP_RESULTS = []
SAVED_MODELS = {}

# -----------------------------
# Utils
# -----------------------------
def load_year_csvs(data_dir, years):
    paths = []
    for y in years:
        pattern = os.path.join(data_dir, f"*{y}*.csv")
        paths.extend(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs found under {data_dir} for years {years}")
    dfs = []
    for p in sorted(paths):
        df = pd.read_csv(p, low_memory=False)
        df["__source_file"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def basic_clean(df):
    df.columns = [c.strip().lower() for c in df.columns]

    for c in [COL_OPEN, COL_CLOSE]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    if COL_SUBJ in df.columns:
        df[COL_SUBJ] = df[COL_SUBJ].fillna("")

    df = df.dropna(subset=[COL_OPEN, COL_CLOSE]).copy()
    df["duration_hours"] = (df[COL_CLOSE] - df[COL_OPEN]).dt.total_seconds() / 3600.0
    df = df[(df["duration_hours"] >= 0) & np.isfinite(df["duration_hours"])]

    if COL_ID in df.columns:
        df = df.sort_values([COL_ID, COL_OPEN]).drop_duplicates(subset=[COL_ID], keep="first")

    if all(c in df.columns for c in [COL_LAT, COL_LON]):
        df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
        df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
        good_geo = (
            (df[COL_LAT].between(42.0, 43.0, inclusive="both")) &
            (df[COL_LON].between(-71.5, -70.5, inclusive="both"))
        )
        df.loc[~good_geo, [COL_LAT, COL_LON]] = np.nan

        if USE_SPATIAL:
            df["dist_city_hall_km"] = np.sqrt(
                (df[COL_LAT] - CITY_HALL_LAT) ** 2 +
                (df[COL_LON] - CITY_HALL_LON) ** 2
            ) * 111.0

    df["open_year"]  = df[COL_OPEN].dt.year
    df["open_month"] = df[COL_OPEN].dt.month
    df["open_dow"]   = df[COL_OPEN].dt.dayofweek
    df["open_hour"]  = df[COL_OPEN].dt.hour
    df["is_weekend"] = df["open_dow"].isin([5, 6]).astype(int)
    df["open_date"]  = df[COL_OPEN].dt.floor("D")

    cap = df["duration_hours"].quantile(0.99)
    df["duration_hours_winsor"] = np.clip(df["duration_hours"], 0, cap)
    return df

def add_simple_rolling_volume(df):
    tmp = df[["open_date"]].copy()
    daily = tmp.groupby("open_date").size().rename("daily_volume").to_frame()
    daily["daily_vol_roll7"] = daily["daily_volume"].rolling(7, min_periods=1).mean()
    df = df.merge(daily, on="open_date", how="left")
    return df

def add_holiday_flag(df):
    if USE_HOLIDAYS:
        df["is_holiday"] = df["open_date"].dt.date.apply(lambda d: d in US_HOLIDAYS).astype(int)
    return df

def load_weather():
    path = os.path.join(DATA_DIR, "boston_hourly_2018_2025.csv")
    w = pd.read_csv(path, low_memory=False)

    if "time" in w.columns:
        w["timestamp"] = pd.to_datetime(w["time"], utc=True)
        w = w.drop(columns=["time"])
    elif "date" in w.columns:
        w["timestamp"] = pd.to_datetime(w["date"], utc=True)
        w = w.drop(columns=["date"])

    keep = ["timestamp"] + [c for c in WEATHER_COLS if c in w.columns]
    w = w[keep]
    return w

def merge_weather(df, weather_df):
    df = df.copy()
    df["open_hour_floor"] = df[COL_OPEN].dt.floor("H")

    w = weather_df.copy()
    w["open_hour_floor"] = pd.to_datetime(w["timestamp"], utc=True)
    w = w.drop(columns=["timestamp"])

    df = df.merge(w, on="open_hour_floor", how="left")
    df.drop(columns=["open_hour_floor"], inplace=True)
    return df

def temporal_splits(df):
    train = df[df["open_year"].between(min(YEARS_TO_USE), VAL_YEAR - 1)]
    val   = df[df["open_year"] == VAL_YEAR]
    test  = df[df["open_year"].isin(TEST_YEARS)]
    return train.copy(), val.copy(), test.copy()

def rmse_compat(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def metrics(y_true_winsor, y_pred):
    return {
        "MAE":   float(np.abs(y_true_winsor - y_pred).mean()),
        "RMSE":  float(rmse_compat(y_true_winsor, y_pred)),
        "MedAE": float(np.median(np.abs(y_true_winsor - y_pred))),
    }

def log_results(model_name, split_name, metrics_dict, extra=None):
    row = {
        "model": model_name,
        "split": split_name,
    }
    row.update(metrics_dict)
    if extra is not None:
        row.update(extra)
    EXP_RESULTS.append(row)

def save_predictions_df(model_name, split_name, df_split, y_true_w, y_pred):
    if COL_ID in df_split.columns:
        ids = df_split[COL_ID].values
    else:
        ids = np.arange(len(df_split))
    out = pd.DataFrame({
        "case_id": ids,
        "open_dt": df_split[COL_OPEN].values,
        "duration_hours_winsor_true": y_true_w,
        "duration_hours_pred": y_pred,
    })
    fname = f"predictions_{model_name.lower()}_{split_name}.csv"
    out.to_csv(os.path.join(REPORT_DIR, fname), index=False)

# -----------------------------
# EDA
# -----------------------------
def plot_duration_hist(df):
    plt.figure()
    df["duration_hours"].clip(upper=df["duration_hours"].quantile(0.99)).plot.hist(bins=60)
    plt.xlabel("Duration to Close (hours) — clipped @99th pct")
    plt.ylabel("Count")
    plt.title("Distribution of Resolution Time (Hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_duration_hist.png"), dpi=160)
    plt.close()

def plot_box_by_reason(df, top_n=10):
    if COL_REASON not in df.columns:
        return
    top_reasons = df[COL_REASON].value_counts().head(top_n).index.tolist()
    sub = df[df[COL_REASON].isin(top_reasons)].copy()
    sub["duration_clip"] = sub["duration_hours"].clip(upper=df["duration_hours"].quantile(0.99))
    plt.figure(figsize=(10, 5))
    sub.boxplot(column="duration_clip", by=COL_REASON)
    plt.suptitle("")
    plt.title(f"Duration by Top-{top_n} Reasons")
    plt.xlabel("Reason")
    plt.ylabel("Hours to Close (clipped)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_box_by_reason.png"), dpi=160)
    plt.close()

def plot_monthly_trend(df):
    m = (
        df.set_index(COL_OPEN)["duration_hours"]
          .clip(upper=df["duration_hours"].quantile(0.99))
          .resample("MS").median()
    )
    plt.figure()
    m.plot()
    plt.ylabel("Monthly Median Hours to Close (clipped)")
    plt.title("Trend of Median Resolution Time by Month")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_monthly_median_trend.png"), dpi=160)
    plt.close()

class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

# -----------------------------
# Interpretability helpers
# -----------------------------
def get_feature_names_from_preprocessor(pre):
    feature_names = []

    for name, transformer, cols in pre.transformers_:
        if name == "cat":
            ohe = transformer.named_steps.get("ohe", None)
            if ohe is not None:
                ohe_names = ohe.get_feature_names_out(cols)
                feature_names.extend(ohe_names.tolist())
        elif name == "num":
            feature_names.extend(list(cols))
        elif name == "txt":
            tfidf = transformer
            if isinstance(transformer, Pipeline):
                tfidf = transformer.named_steps.get("tfidf", None)
            if tfidf is not None:
                txt_names = tfidf.get_feature_names_out()
                feature_names.extend([f"tfidf_{t}" for t in txt_names])

    return feature_names

def plot_feature_importance_hgbr(pipe, train, top_n=20, sample_size=50000):
    pre   = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    if len(train) > sample_size:
        train_sample = train.sample(sample_size, random_state=42).copy()
    else:
        train_sample = train.copy()

    X = pre.transform(train_sample)
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = train_sample["duration_hours_winsor"].values

    print("Computing permutation importances for HGBR (this may take a bit)…")
    result = permutation_importance(
        model, X, y,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importances = result.importances_mean

    feat_names = get_feature_names_from_preprocessor(pre)
    if len(feat_names) != len(importances):
        feat_names = [f"f{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.xlabel("Permutation importance (Δ error)")
    plt.title(f"HGBR Top {top_n} Feature Importances")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "04_hgbr_feature_importance.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_manual_pdp(pipe, df, feature, n_points=20, sample_size=20000):
    if feature not in df.columns:
        return

    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()

    col = df_sample[feature]
    if pd.api.types.is_numeric_dtype(col):
        vmin, vmax = col.quantile(0.01), col.quantile(0.99)
        grid = np.linspace(vmin, vmax, n_points)
    else:
        grid = sorted(col.unique())

    mean_preds = []
    for v in grid:
        df_tmp = df_sample.copy()
        df_tmp[feature] = v
        preds = np.expm1(pipe.predict(df_tmp))
        mean_preds.append(preds.mean())

    plt.figure()
    plt.plot(grid, mean_preds, marker="o")
    plt.xlabel(feature)
    plt.ylabel("Predicted hours to close")
    plt.title(f"Approx. partial dependence: {feature}")
    plt.tight_layout()
    safe_feature = str(feature).replace("/", "_")
    out_path = os.path.join(FIG_DIR, f"05_pdp_{safe_feature}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

def run_pdp_suite(pipe, train):
    for feat in ["open_hour", "open_dow", "daily_vol_roll7", "is_weekend"]:
        if feat in train.columns:
            plot_manual_pdp(pipe, train, feat)

def plot_residual_diagnostics(pipe, test, split_name="test"):
    y_true = test["duration_hours_winsor"].values
    y_pred = np.expm1(pipe.predict(test))

    cap = np.quantile(y_true, 0.99)
    y_true_clip = np.clip(y_true, 0, cap)
    y_pred_clip = np.clip(y_pred, 0, cap)

    residuals = y_true_clip - y_pred_clip

    plt.figure()
    plt.hist(residuals, bins=60)
    plt.xlabel("Residual (true - pred) hours")
    plt.ylabel("Count")
    plt.title(f"Residual distribution ({split_name} set, clipped @99th pct)")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"06_residual_hist_{split_name}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

    n = len(y_true_clip)
    max_points = 40000
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        yt_s = y_true_clip[idx]
        yp_s = y_pred_clip[idx]
    else:
        yt_s, yp_s = y_true_clip, y_pred_clip

    plt.figure()
    plt.scatter(yt_s, yp_s, s=2, alpha=0.3)
    plt.plot([0, cap], [0, cap], color="red", linewidth=1)
    plt.xlabel("True hours to close (clipped)")
    plt.ylabel("Predicted hours to close (clipped)")
    plt.title(f"True vs Predicted ({split_name} set)")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f"07_true_vs_pred_{split_name}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()

def run_interpretability(train, test):
    if "HGBR" not in SAVED_MODELS:
        print("No HGBR model found in SAVED_MODELS; skipping interpretability.")
        return

    pipe = SAVED_MODELS["HGBR"]

    print("Generating HGBR feature importance and interpretability plots…")
    plot_feature_importance_hgbr(pipe, train)
    run_pdp_suite(pipe, train)
    plot_residual_diagnostics(pipe, test, split_name="test")
    print(f"Interpretability figures saved to: {FIG_DIR}")

# -----------------------------
# Features / Modeling
# -----------------------------
def build_feature_pipeline(df):
    def make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=50)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_cols = [c for c in [COL_REASON, COL_TYPE, COL_SRC, COL_NHOOD] if c in df.columns]

    num_base = [
        "open_hour",
        "open_dow",
        "open_month",
        "is_weekend",
        "daily_vol_roll7",
    ]
    if USE_HOLIDAYS and "is_holiday" in df.columns:
        num_base.append("is_holiday")
    if USE_SPATIAL and "dist_city_hall_km" in df.columns:
        num_base.append("dist_city_hall_km")

    if USE_WEATHER:
        for col in WEATHER_COLS:
            if col in df.columns:
                num_base.append(col)

    num_cols = [c for c in num_base if c in df.columns]

    transformers = []

    if cat_cols:
        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe()),
            ]
        )
        transformers.append(("cat", cat_transformer, cat_cols))

    if num_cols:
        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_transformer, num_cols))

    if USE_TEXT and (COL_SUBJ in df.columns):
        transformers.append(
            ("txt", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), COL_SUBJ)
        )

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

def fit_and_eval(train, val, test):
    y_train_w = train["duration_hours_winsor"].values
    y_val_w   = val["duration_hours_winsor"].values
    y_tst_w   = test["duration_hours_winsor"].values

    y_train_log = np.log1p(y_train_w)

    pre = build_feature_pipeline(train)

    if FAST_MODE:
        model_dict = {"Ridge": Ridge(alpha=1.0, random_state=42)}
    else:
        model_dict = {
            "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
            "LassoCV": LassoCV(alphas=np.logspace(-3, 1, 9), max_iter=10000),
            "HGBR": HistGradientBoostingRegressor(
                max_depth=None,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            ),
        }

    pred_cap = train["duration_hours"].quantile(PRED_CAP_PCT)

    for name, model in model_dict.items():
        if name == "HGBR":
            pipe = Pipeline([
               ("pre", pre),
               ("to_dense", ToDenseTransformer()),
               ("model", model),
            ])
        else:
            pipe = Pipeline([
               ("pre", pre),
               ("model", model),
            ])

        pipe.fit(train, y_train_log)
        SAVED_MODELS[name] = pipe

        pred_val_log  = pipe.predict(val)
        pred_test_log = pipe.predict(test)

        pred_val  = np.expm1(pred_val_log)
        pred_test = np.expm1(pred_test_log)

        pred_val  = np.clip(pred_val,  0, pred_cap)
        pred_test = np.clip(pred_test, 0, pred_cap)

        r_val  = metrics(y_val_w,  pred_val)
        r_test = metrics(y_tst_w, pred_test)

        print(f"\n{name}")
        print(f"  VAL {VAL_YEAR}      → MAE: {r_val['MAE']:.3f}, RMSE: {r_val['RMSE']:.3f}, MedAE: {r_val['MedAE']:.3f}")
        print(f"  TEST {TEST_YEARS} → MAE: {r_test['MAE']:.3f}, RMSE: {r_test['RMSE']:.3f}, MedAE: {r_test['MedAE']:.3f}")

        log_results(name, "val",  r_val)
        log_results(name, "test", r_test)

        save_predictions_df(name, "val",  val,  y_val_w,  pred_val)
        save_predictions_df(name, "test", test, y_tst_w, pred_test)

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        import sklearn
        print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    except Exception:
        pass
    print(f"USE_TEXT={USE_TEXT}, USE_WEATHER={USE_WEATHER}, USE_HOLIDAYS={USE_HOLIDAYS}, USE_SPATIAL={USE_SPATIAL}")
    print(f"FAST_MODE={FAST_MODE}, PRED_CAP_PCT={PRED_CAP_PCT}")

    print("Loading CSVs…")
    df = load_year_csvs(DATA_DIR, YEARS_TO_USE)
    print(f"Loaded {len(df):,} rows")

    print("Cleaning & feature engineering…")
    df = basic_clean(df)
    df = add_simple_rolling_volume(df)
    df = add_holiday_flag(df)

    if USE_WEATHER:
        weather_df = load_weather()
        if weather_df is None:
            print("⚠️ USE_WEATHER=True but load_weather() returned None. Skipping weather merge.")
        else:
            df = merge_weather(df, weather_df)

    print("Saving preliminary figures…")
    plot_duration_hist(df)
    plot_box_by_reason(df, top_n=10)
    plot_monthly_trend(df)
    print(f"Figures saved to: {FIG_DIR}")

    train, val, test = temporal_splits(df)
    print(f"Train: {len(train):,} | Val({VAL_YEAR}): {len(val):,} | Test({TEST_YEARS}): {len(test):,}")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("⚠️ One of the splits is empty. Check uploaded years or split settings.")
        return

    print("Fitting models and reporting results…")
    fit_and_eval(train, val, test)

    run_interpretability(train, test)

    if EXP_RESULTS:
        exp_path = os.path.join(REPORT_DIR, "experiment_results.csv")
        pd.DataFrame(EXP_RESULTS).to_csv(exp_path, index=False)
        print(f"\nExperiment results saved to: {exp_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
