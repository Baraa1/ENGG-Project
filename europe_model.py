"""
europe_model.py - Europe regional model (ENGG2112 project, Nhi's part).

Goal: predict annual PM2.5 exposure for European countries from the emitting
share of electricity generation, country and year (proposal section 2.2), so
that it can be compared against the world model and the other regional models.

Design decisions (all follow the proposal / Kelly's message):
  * Training data: 2000-2018.  Final test: 2019 to the latest year (2022).
    The test years are split off first and never used for fitting, scaling,
    model selection or cross validation.
  * Cross validation: forward-chaining (expanding window) over years inside the
    training period, so validation years are always later than the training
    years, the same situation as the final test.
  * Target is z-scored with training-set statistics (TransformedTargetRegressor),
    numeric features are scaled inside the pipeline, so every scaler is fitted
    on training data only (on the CV training part during CV).
  * R2, MAE and RMSE are reported separately (never combined into one score).
    MAE/RMSE are given in ug/m3 and in normalised (z) units.
  * Models are chosen by CV RMSE only.  Test metrics of every candidate are also
    printed for documentation, but they are not used to choose the model.

Run from the repo root:   python europe_model.py
Outputs go to:            outputs/europe/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "cleaned_climate_merged.csv"
OUT_DIR = HERE / "outputs" / "europe"

TRAIN_START = 2000
TRAIN_END = 2018
TEST_START = 2019  # test = TEST_START .. latest year in the data
RANDOM_STATE = 42

# Forward-chaining CV folds: (last training year, first validation year, last validation year)
CV_FOLDS = [(2008, 2009, 2012), (2012, 2013, 2015), (2015, 2016, 2018)]

# Our definition of "Europe" - keep this identical across the team's notes.
# Included on purpose: Russia, Turkey, Cyprus (judgement calls).
# Excluded on purpose: Georgia, Armenia, Azerbaijan, Kazakhstan, Greenland.
EUROPE = [
    "Albania", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
    "Portugal", "Romania", "Russia", "Serbia", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom",
]

# Feature sets: which columns each candidate model may use.
FEATURE_SETS = {
    "emit only": {"num": ["emit"], "cat": []},
    "emit + year": {"num": ["emit", "year"], "cat": []},
    "emit + year + country": {"num": ["emit", "year"], "cat": ["country"]},
    # ablation: same as above WITHOUT the emitting share, to see what emit adds.
    # It is a control for the report, not an eligible final model (the proposal's
    # model predicts PM2.5 from emitting share, country and year).
    "year + country (no emit)": {"num": ["year"], "cat": ["country"]},
}
ABLATION_SET = "year + country (no emit)"


def load_europe():
    df = pd.read_csv(DATA_FILE)
    expected = ["country", "year", "% GHG emitting source",
                "% GHG non emitting source", "PM2.5 Exposure"]
    if list(df.columns) != expected:
        raise SystemExit(
            f"{DATA_FILE.name} has unexpected columns: {list(df.columns)}\n"
            "It is probably a corrupted copy (e.g. re-saved by Excel with ';' separators "
            "and mangled decimals).\nRestore the original with:\n"
            "  git checkout main -- cleaned_climate_merged.csv"
        )
    df = df.rename(columns={
        "% GHG emitting source": "emit",
        "% GHG non emitting source": "nonemit",
        "PM2.5 Exposure": "pm",
    })
    missing = [c for c in EUROPE if c not in set(df["country"])]
    if missing:
        print(f"WARNING: not in the data (dropped from Europe list): {missing}")
    eu = df[df["country"].isin(EUROPE)].copy()
    return eu.sort_values(["country", "year"]).reset_index(drop=True)


def split_train_test(eu):
    train = eu[(eu["year"] >= TRAIN_START) & (eu["year"] <= TRAIN_END)].reset_index(drop=True)
    test = eu[eu["year"] >= TEST_START].reset_index(drop=True)
    assert train["year"].max() < test["year"].min(), "train/test overlap in time"
    return train, test


def make_cv_splits(train):
    splits = []
    for last_train, val_start, val_end in CV_FOLDS:
        tr_idx = np.where(train["year"] <= last_train)[0]
        va_idx = np.where((train["year"] >= val_start) & (train["year"] <= val_end))[0]
        splits.append((tr_idx, va_idx))
    return splits


def make_pipeline(features, estimator):
    transformers = [("num", StandardScaler(), features["num"])]
    if features["cat"]:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), features["cat"]))
    pre = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("pre", pre), ("est", estimator)])
    # z-score the target using training statistics only
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def metrics(y_true, y_pred, y_train_std):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mae,
        "RMSE": rmse,
        "MAE_z": mae / y_train_std,
        "RMSE_z": rmse / y_train_std,
    }


def baseline_predictions(train, test):
    """Simple reference predictions the real models should beat."""
    mean_pred = np.full(len(test), train["pm"].mean())
    # persistence: each country's most recent training-period value
    last = train.sort_values("year").groupby("country")["pm"].last()
    persist_pred = test["country"].map(last).to_numpy()
    return {
        "baseline: training mean": mean_pred,
        "baseline: last training year (persistence)": persist_pred,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eu = load_europe()
    train, test = split_train_test(eu)
    y_train_std = train["pm"].std()

    print("=" * 70)
    print("EUROPE MODEL")
    print("=" * 70)
    print(f"Countries in Europe list : {eu['country'].nunique()}")
    print(f"Train rows ({TRAIN_START}-{TRAIN_END}) : {len(train)}")
    print(f"Test rows  ({TEST_START}-{eu['year'].max()}) : {len(test)}")
    print(f"Train PM2.5: mean {train['pm'].mean():.2f}, std {y_train_std:.2f} ug/m3")
    print(f"Test  PM2.5: mean {test['pm'].mean():.2f}, std {test['pm'].std():.2f} ug/m3")
    unseen = sorted(set(test["country"]) - set(train["country"]))
    print(f"Test countries never seen in training: {unseen if unseen else 'none'}")

    cv_splits = make_cv_splits(train)
    for i, (tr, va) in enumerate(cv_splits, 1):
        print(f"CV fold {i}: train years <= {train['year'].iloc[tr].max()} ({len(tr)} rows), "
              f"validate {train['year'].iloc[va].min()}-{train['year'].iloc[va].max()} ({len(va)} rows)")

    X_cols = ["country", "year", "emit"]
    X_train, y_train = train[X_cols], train["pm"]
    X_test, y_test = test[X_cols], test["pm"]

    models = {
        "Ridge": (Ridge(), {"regressor__est__alpha": [0.1, 1, 10, 100]}),
        "RandomForest": (
            RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
            {
                "regressor__est__max_depth": [None, 6, 12],
                "regressor__est__min_samples_leaf": [1, 3, 5],
            },
        ),
    }

    rows = []
    fitted = {}
    for feat_name, features in FEATURE_SETS.items():
        for model_name, (estimator, grid) in models.items():
            search = GridSearchCV(
                make_pipeline(features, estimator),
                param_grid=grid,
                cv=cv_splits,
                scoring="neg_root_mean_squared_error",
                refit=True,
            )
            search.fit(X_train, y_train)
            pred = search.predict(X_test)
            key = f"{model_name} | {feat_name}"
            fitted[key] = (search, pred)
            params = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
            rows.append({
                "model": key,
                "cv_RMSE": -search.best_score_,
                "best_params": str(params),
                **{f"test_{k}": v for k, v in metrics(y_test, pred, y_train_std).items()},
            })

    for name, pred in baseline_predictions(train, test).items():
        rows.append({
            "model": name, "cv_RMSE": np.nan, "best_params": "-",
            **{f"test_{k}": v for k, v in metrics(y_test, pred, y_train_std).items()},
        })

    results = pd.DataFrame(rows)
    # eligible = fitted models that use the emitting share (the proposal's model);
    # the no-emit ablation and the baselines are shown for comparison only.
    eligible = results["model"].isin(fitted) & ~results["model"].str.contains(ABLATION_SET, regex=False)
    best_key = results[eligible].sort_values("cv_RMSE").iloc[0]["model"]
    results["selected_by_cv"] = results["model"] == best_key
    results = results.sort_values(["selected_by_cv", "cv_RMSE"], ascending=[False, True])
    results.to_csv(OUT_DIR / "europe_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    show = results.drop(columns=["best_params"]).round(3)
    print("\n" + "=" * 70)
    print("RESULTS  (selected by CV RMSE only; test columns are the 2019+ hold-out)")
    print("=" * 70)
    print(show.to_string(index=False))
    print(f"\nSelected model (lowest CV RMSE among models using emitting share): {best_key}")
    ablation_rows = results[results["model"].str.contains(ABLATION_SET, regex=False)]
    print("Ablation (no emitting share) test R2 by model:",
          {r["model"].split(" |")[0]: round(r["test_R2"], 3) for _, r in ablation_rows.iterrows()})
    print("-> If these match or beat the selected model, emitting share adds little "
          "once country and year are known.")
    print("Best params:", results[results['model'] == best_key]['best_params'].iloc[0])

    # Save test predictions of the selected model so they can be combined/compared later
    best_search, best_pred = fitted[best_key]
    out = test[["country", "year", "pm"]].rename(columns={"pm": "actual"})
    out["predicted"] = best_pred
    out.to_csv(OUT_DIR / "europe_test_predictions.csv", index=False)

    final = metrics(y_test, best_pred, y_train_std)
    print("\nFINAL TEST (selected model, 2019+ hold-out, evaluated once):")
    print(f"  R2   : {final['R2']:.4f}")
    print(f"  MAE  : {final['MAE']:.3f} ug/m3   ({final['MAE_z']:.3f} normalised)")
    print(f"  RMSE : {final['RMSE']:.3f} ug/m3   ({final['RMSE_z']:.3f} normalised)")

    # Plot: actual vs predicted on the hold-out
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(out["actual"], out["predicted"], alpha=0.6)
    lo = min(out["actual"].min(), out["predicted"].min())
    hi = max(out["actual"].max(), out["predicted"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect prediction")
    ax.set_xlabel("Actual PM2.5 (ug/m3)")
    ax.set_ylabel("Predicted PM2.5 (ug/m3)")
    ax.set_title(f"Europe hold-out {TEST_START}-{eu['year'].max()}: {best_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "europe_actual_vs_predicted.png", dpi=150)
    print(f"\nSaved results, predictions and plot to {OUT_DIR}")


if __name__ == "__main__":
    main()
