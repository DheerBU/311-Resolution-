# midterm_pipeline.py
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
from sklearn.ensemble import GradientBoostingRegressor

# -----------------------------
# Config (YOUR PATHS + toggles)
# -----------------------------
DATA_DIR = "/Users/dheerdoshi/Documents/CS506-proj/data"          # CSVs (2018–2025) here
FIG_DIR  = "/Users/dheerdoshi/Documents/CS506-proj/report/figures" # figures saved here
os.makedirs(FIG_DIR, exist_ok=True)

YEARS_TO_USE = list(range(2018, 2026)) 
VAL_YEAR     = 2023
TEST_YEARS   = [2024, 2025]    

# Speed / stability toggles
USE_TEXT       = False        # turn on later after baseline is stable
FAST_MODE      = True         # run a fast Ridge baseline only
PRED_CAP_PCT   = 0.99         # clip predictions at train 99th percentile to avoid blow-ups

# Expected column names (adjust if your files differ)
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

    # parse datetimes
    for c in [COL_OPEN, COL_CLOSE]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # drop invalid timestamps and negative durations
    df = df.dropna(subset=[COL_OPEN, COL_CLOSE]).copy()
    df["duration_hours"] = (df[COL_CLOSE] - df[COL_OPEN]).dt.total_seconds() / 3600.0
    df = df[(df["duration_hours"] >= 0) & np.isfinite(df["duration_hours"])]

    # dedupe by case_id if present
    if COL_ID in df.columns:
        df = df.sort_values([COL_ID, COL_OPEN]).drop_duplicates(subset=[COL_ID], keep="first")

    # validate lat/lon (optional; keep NaN if outside)
    if all(c in df.columns for c in [COL_LAT, COL_LON]):
        df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
        df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")
        good_geo = (
            (df[COL_LAT].between(42.0, 43.0, inclusive="both")) &
            (df[COL_LON].between(-71.5, -70.5, inclusive="both"))
        )
        df.loc[~good_geo, [COL_LAT, COL_LON]] = np.nan

    # temporal features
    df["open_year"]  = df[COL_OPEN].dt.year
    df["open_month"] = df[COL_OPEN].dt.month
    df["open_dow"]   = df[COL_OPEN].dt.dayofweek  # 0=Mon
    df["open_hour"]  = df[COL_OPEN].dt.hour
    df["is_weekend"] = df["open_dow"].isin([5, 6]).astype(int)

    # winsorize 99th pct for label
    cap = df["duration_hours"].quantile(0.99)
    df["duration_hours_winsor"] = np.clip(df["duration_hours"], 0, cap)
    return df

def add_simple_rolling_volume(df):
    """Approx recent workload: 7-day rolling mean of daily volume."""
    tmp = df[[COL_OPEN]].copy()
    tmp["open_date"] = df[COL_OPEN].dt.floor("D")
    daily = tmp.groupby("open_date").size().rename("daily_volume").to_frame()
    daily["daily_vol_roll7"] = daily["daily_volume"].rolling(7, min_periods=1).mean()
    df["open_date"] = df[COL_OPEN].dt.floor("D")
    df = df.merge(daily, on="open_date", how="left")
    df.drop(columns=["open_date"], inplace=True)
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
        "MAE": np.abs(y_true_winsor - y_pred).mean(),
        "RMSE": rmse_compat(y_true_winsor, y_pred),
        "MedAE": np.median(np.abs(y_true_winsor - y_pred)),
    }

# -----------------------------
# EDA (Preliminary Visualizations)
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
    sub["duration_clip"] = sub["duration_hours"].clip(upper=sub["duration_hours"].quantile(0.99))
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

# -----------------------------
# Features / Modeling
# -----------------------------
def build_feature_pipeline(df):
    # robust OHE constructor (backward compatible)
    def make_ohe():
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=50)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_cols = [c for c in [COL_REASON, COL_TYPE, COL_SRC, COL_NHOOD] if c in df.columns]
    num_cols = [c for c in ["open_hour", "open_dow", "open_month", "is_weekend", "daily_vol_roll7"] if c in df.columns]

    transformers = []
    if cat_cols:
        transformers.append(("cat", make_ohe(), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if USE_TEXT and (COL_SUBJ in df.columns):
        transformers.append(("txt", TfidfVectorizer(max_features=3000, ngram_range=(1,2)), COL_SUBJ))

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

def fit_and_eval(train, val, test):
    # train on log1p of winsorized duration
    y_train_log = np.log1p(train["duration_hours_winsor"].values)
    y_val_w     = val["duration_hours_winsor"].values
    y_tst_w     = test["duration_hours_winsor"].values

    pre = build_feature_pipeline(train)

    if FAST_MODE:
        model_dict = {"Ridge": Ridge(alpha=1.0, random_state=42)}
    else:
        model_dict = {
            "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
            "LassoCV": LassoCV(alphas=np.logspace(-3, 1, 9), max_iter=10000),
            "GBR": GradientBoostingRegressor(random_state=42),
        }

    # cap predictions to avoid numerical explosions
    pred_cap = train["duration_hours"].quantile(PRED_CAP_PCT)

    for name, model in model_dict.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(train, y_train_log)

        pred_val  = np.expm1(pipe.predict(val))
        pred_test = np.expm1(pipe.predict(test))

        pred_val  = np.clip(pred_val,  0, pred_cap)
        pred_test = np.clip(pred_test, 0, pred_cap)

        r_val  = metrics(y_val_w,  pred_val)
        r_test = metrics(y_tst_w, pred_test)

        print(f"\n{name}")
        print(f"  VAL {VAL_YEAR}  → MAE: {r_val['MAE']:.3f}, RMSE: {r_val['RMSE']:.3f}, MedAE: {r_val['MedAE']:.3f}")
        print(f"  TEST {TEST_YEARS} → MAE: {r_test['MAE']:.3f}, RMSE: {r_test['RMSE']:.3f}, MedAE: {r_test['MedAE']:.3f}")

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        import sklearn
        print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")
    except Exception:
        pass
    print(f"USE_TEXT={USE_TEXT}, FAST_MODE={FAST_MODE}, PRED_CAP_PCT={PRED_CAP_PCT}")

    print("Loading CSVs…")
    df = load_year_csvs(DATA_DIR, YEARS_TO_USE)
    print(f"Loaded {len(df):,} rows")

    print("Cleaning & feature engineering…")
    df = basic_clean(df)
    df = add_simple_rolling_volume(df)

    print("Saving preliminary figures…")
    plot_duration_hist(df)
    plot_box_by_reason(df, top_n=10)
    plot_monthly_trend(df)
    print(f"Figures saved to: {FIG_DIR}")

    train, val, test = temporal_splits(df)
    print(f"Train: {len(train):,} | Val({VAL_YEAR}): {len(val):,} | Test({TEST_YEARS}): {len(test):,}")

    if len(train)==0 or len(val)==0 or len(test)==0:
        print("⚠️ One of the splits is empty. Check uploaded years or split settings.")
        return

    print("Fitting baselines and reporting preliminary results…")
    fit_and_eval(train, val, test)
    print("\nDone.")

if __name__ == "__main__":
    main()
