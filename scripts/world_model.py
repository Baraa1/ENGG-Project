"""
world_model.py - the world model: predict PM2.5 exposure from electricity generation, using all countries.

Script version of notebooks/02_world_model.ipynb. It runs the same steps with the same settings
and produces the same files, so either can be used. The notebook has the full plain-English
explanations; the section headings here (Part 1 ... Part 7, Step 1b ...) match the notebook's,
so you can look up any step there.

Run (from any folder):  python scripts/world_model.py
Reads:                  data/processed/cleaned_climate_merged.csv, data/raw/EDGAR_*.xlsx,
                        and the World Energy Consumption dataset (downloaded from Kaggle by kagglehub)
Writes to results/:     world_model_metrics.csv, world_model.joblib, ablation_metrics.csv,
                        within_country_metrics.csv and the charts (.png)
"""
import os
from pathlib import Path

import joblib
import kagglehub
import matplotlib

matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


# =====================================================================================
# Step 0 - Settings
# =====================================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in scripts/)
DATA_FILE = ROOT / "data" / "processed" / "cleaned_climate_merged.csv"
ENERGY_DATASET = "pralabhpoudel/world-energy-consumption"   # Kaggle, same source as the cleaning step
EDGAR_FILES = {   # EDGAR v8.1 air pollutant emissions, Gg (thousand tonnes) per year
    "so2": ROOT / "data" / "raw" / "EDGAR_SO2_1970_2022.xlsx",
    "nox": ROOT / "data" / "raw" / "EDGAR_NOx_1970_2022.xlsx",
    "pm25": ROOT / "data" / "raw" / "EDGAR_PM2.5_1970_2022.xlsx",
}
POWER_SECTOR = "1.A.1.a"   # EDGAR/IPCC code for "Main Activity Electricity and Heat Production" it's used to pick out only power-plant rows.
OUT_DIR = ROOT / "results"
TARGET = "PM2.5 Exposure"
SPLIT_YEAR = 2018   # Train on years before this, test on years from it on. This gave 83% / 17%.
N_FOLDS = 5
TEST_COUNTRY = "Australia"

# Original plan: two shares that add up to ~100%.
ORIGINAL_FEATURES = ["% GHG emitting source", "% GHG non emitting source"]
# Change 1 - feature set C: electricity-generation columns from the same energy dataset.
SOURCE_SHARES = ["coal_share_elec", "gas_share_elec", "oil_share_elec", "hydro_share_elec",
                 "nuclear_share_elec", "solar_share_elec", "wind_share_elec", "biofuel_share_elec"]
# Per capita means per person.
PER_PERSON = ["coal_elec_per_capita", "gas_elec_per_capita", "oil_elec_per_capita",
              "fossil_elec_per_capita", "per_capita_electricity"]
# Change 2 - feature set D: C + power-plant emissions from EDGAR.
POWER_EMISSIONS = []   # e.g. "power_so2_per_kwh", "power_so2_per_capita", ...

# per_kwh is how dirty each unit of electricity is
# per_capita is how much power-plant pollution there is relative to the population.
for pollutant in EDGAR_FILES:
    for unit in ["per_kwh", "per_capita"]:
        POWER_EMISSIONS.append(f"power_{pollutant}_{unit}")

FEATURE_SETS = {
    "Original (2 shares)": ORIGINAL_FEATURES,
    "C (shares + per-person)": SOURCE_SHARES + PER_PERSON,
    "D (C + power-plant emissions)": SOURCE_SHARES + PER_PERSON + POWER_EMISSIONS,
}

# Regions/income groups in the source data. They overlap with the individual countries,
# so keeping them would double-count in a "world" model.
AGGREGATES = [
    "World", "European Union (27)", "High-income countries", "Upper-middle-income countries",
    "Lower-middle-income countries", "Low-income countries",
]

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================================
# Part 1 - Look at the data first
# =====================================================================================
section("Part 1 - Load and inspect the data")
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")
# .nunique() counts unique values, so it gives the number of distinct countries rather than the number of rows.
print(f"Years: {df.year.min()}-{df.year.max()}, countries/regions: {df.country.nunique()}")
print(f"Missing values per column:\n{df.isna().sum()}\n")

# drop the Regions/income columns from the source data
df = df[~df["country"].isin(AGGREGATES)].reset_index(drop=True)
print(f"After dropping aggregates: {df.shape[0]} rows, {df.country.nunique()} countries")

# ---- Step 1b - Add the extra electricity columns for feature set C (change 1)
energy_path = kagglehub.dataset_download(ENERGY_DATASET)
energy = pd.read_csv(os.path.join(energy_path, "World Energy Consumption.csv"))

energy_cols = SOURCE_SHARES + PER_PERSON + ["iso_code", "population", "electricity_generation"]
n_before = len(df)
df = df.merge(energy[["country", "year"] + energy_cols], on=["country", "year"], how="left")
assert len(df) == n_before, "merge should not add or drop rows"

print("\nMissing values in the new columns (% of rows):")
print((df[SOURCE_SHARES + PER_PERSON].isna().mean() * 100).round(1).to_string())


# ---- Step 1c - Add power-plant emissions for feature set D (change 2)
def power_sector_emissions(path, name):
    """One pollutant's yearly power-sector emissions per country (Gg), from an EDGAR spreadsheet."""
    sheet = pd.read_excel(path, sheet_name="IPCC 2006", header=9)   # first 9 rows are a title block
    power = sheet[sheet["ipcc_code_2006_for_standard_report"] == POWER_SECTOR]
    # collects every column whose name starts with "Y_". str(c) guards against non-text column names
    year_cols = [c for c in power.columns if str(c).startswith("Y_")]

    # Wide format has one column per year. Long format has one row per country-year, which is the shape our main table df has. .melt converts between them.
    # id_vars are the columns that stay as identifiers.
    # value_vars are the columns to unpivot.
    # var_name is the new column that holds the old column names.
    # value_name is the new column that holds the numbers, here power_so2_gg and so on.
    long = power.melt(id_vars="Country_code_A3", value_vars=year_cols, var_name="year", value_name=name)
    # dropping Y_
    long["year"] = long["year"].str[2:].astype(int)
    # add the fossil and biofuel rows together for each country-year (because we dont't care about fuel type, the clusters are pollutatnt type!)
    return (long.groupby(["Country_code_A3", "year"], as_index=False)[name].sum(min_count=1)
                .rename(columns={"Country_code_A3": "iso_code"}))


generation_kwh = df["electricity_generation"] * 1e9   # TWh -> kWh
for pollutant, path in EDGAR_FILES.items():
    gg = f"power_{pollutant}_gg"
    df = df.merge(power_sector_emissions(path, gg), on=["iso_code", "year"], how="left")
    df[f"power_{pollutant}_per_kwh"] = (df[gg] * 1e9 / generation_kwh.where(generation_kwh > 0))  # g per kWh
    df[f"power_{pollutant}_per_capita"] = df[gg] * 1e6 / df["population"]                       # kg per person
assert len(df) == n_before, "merge should not add or drop rows"

print("\nMissing values in the new columns (% of rows):")
print((df[POWER_EMISSIONS].isna().mean() * 100).round(1).to_string())
print(df[POWER_EMISSIONS].describe().round(2))

# ---- Step 1d - How much of PM2.5 actually comes from power plants?
all_sectors = pd.read_excel(EDGAR_FILES["pm25"], sheet_name="IPCC 2006", header=9)
sector_name = "ipcc_code_2006_for_standard_report_name"
YEAR_COL = f"Y_{SPLIT_YEAR - 1}"   # the last training year

world = all_sectors.groupby(sector_name)[YEAR_COL].sum().sort_values(ascending=False)
print(f"\nWorld PM2.5 emissions by sector, {YEAR_COL[2:]} (% of total):")
print((world / world.sum() * 100).round(1).head(6).to_string())

# .unstack. Grouping by two columns gives a Series with a two-level label (country, sector).
# .unstack moves the inner level (sector) into columns
by_country = all_sectors.groupby(["Country_code_A3", sector_name])[YEAR_COL].sum().unstack(fill_value=0)

# calcualtes the percentage
power_share = by_country["Main Activity Electricity and Heat Production"] / by_country.sum(axis=1) * 100
print(f"\nPower plants' share of a country's PM2.5 emissions: median country {power_share.median():.1f}%, "
      f"Australia {power_share.get('AUS', np.nan):.1f}%")


# =====================================================================================
# Part 2 - Splitting the data by time
# =====================================================================================
section("Part 2 - Split by time")
train = df[df["year"] < SPLIT_YEAR]
test = df[df["year"] >= SPLIT_YEAR]   # touched exactly once, at the very end
# groups_train is each row's country name. It isn't an input to any model.
# It's the label that GroupKFold uses in Step 3b to keep a country's rows together.
y_train, groups_train = train[TARGET], train["country"]
y_test = test[TARGET]

print(f"Train: {len(train)} rows (years {train.year.min()}-{train.year.max()}), {len(train) / len(df):.0%} of the data")
print(f"Test:  {len(test)} rows (years {test.year.min()}-{test.year.max()}), {len(test) / len(df):.0%} of the data")


# =====================================================================================
# Part 3 - Comparing models fairly
# =====================================================================================
section("Part 3 - Cross-validation on the training set")


# ---- Step 3a - The candidate models (gap filling and scaling live inside the Pipeline, so no leakage)
def make_pipeline(estimator):
    return Pipeline([
        ("fill_gaps", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


models = {
    "LinearRegression": make_pipeline(LinearRegression()),
    "Ridge(alpha=1)": make_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    "Ridge(alpha=10)": make_pipeline(Ridge(alpha=10.0, random_state=RANDOM_STATE)),
    "Ridge(alpha=100)": make_pipeline(Ridge(alpha=100.0, random_state=RANDOM_STATE)),
    "RandomForest(depth=5)": make_pipeline(
        RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)),
    "RandomForest(depth=10)": make_pipeline(
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
}

# ---- Step 3b - Cross-validation (on the training set only)
# GroupKFold keeps every country entirely inside one fold, so a country's neighbouring years
# can't leak between train and validation (it is deterministic, so no random_state needed).
cv = GroupKFold(n_splits=N_FOLDS)
scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}


def cross_validate_all(frame, target, feature_sets, candidate_models):
    """CV every model on every feature set; returns a results table and the per-fold scores."""
    rows, folds = [], {}
    for fs_name, features in feature_sets.items():
        for model_name, pipe in candidate_models.items():
            res = cross_validate(pipe, frame[features], frame[target], groups=frame["country"],
                                 cv=cv, scoring=scoring)
            folds[f"{model_name} | {fs_name}"] = {"R2": res["test_r2"], "RMSE": -res["test_rmse"],
                                                  "MAE": -res["test_mae"]}
            rows.append({
                "model": model_name, "feature_set": fs_name,
                "cv_r2_mean": res["test_r2"].mean(), "cv_r2_std": res["test_r2"].std(),
                "cv_rmse_mean": -res["test_rmse"].mean(), "cv_rmse_std": res["test_rmse"].std(),
                "cv_mae_mean": -res["test_mae"].mean(), "cv_mae_std": res["test_mae"].std(),
            })
    # converts a list of dictionaries into a table, with the dictionary keys as column names.
    return pd.DataFrame(rows), folds


results, cv_folds = cross_validate_all(train, TARGET, FEATURE_SETS, models)
print(f"{N_FOLDS}-fold GroupKFold (by country) CV results on the training set:")
print(results.round(3).to_string(index=False))

# ---- Step 3c - Choose the best model (lowest CV RMSE; never by test score)
best = results.loc[results["cv_rmse_mean"].idxmin()]
best_name, best_fs = best["model"], best["feature_set"]
best_features = FEATURE_SETS[best_fs]
print(f"\nBest by CV RMSE: {best_name} with feature set '{best_fs}'")
print(f"Features used: {best_features}")


# =====================================================================================
# Part 4 - Retrain on all training data and test once
# =====================================================================================
section("Part 4 - Final test (used once)")
X_train, X_test = train[best_features], test[best_features]
best_model = clone(models[best_name]).fit(X_train, y_train)
y_pred = best_model.predict(X_test)


def metrics(y_true, y_hat):
    return {
        "r2": r2_score(y_true, y_hat),
        "rmse": np.sqrt(mean_squared_error(y_true, y_hat)),
        "mae": mean_absolute_error(y_true, y_hat),
    }


test_m = metrics(y_test, y_pred)
aus_mask = (test["country"] == TEST_COUNTRY).to_numpy()
aus_m = metrics(y_test[aus_mask], y_pred[aus_mask])

print(f"Test set ({len(test)} rows):  " + ", ".join(f"{k}={v:.4f}" for k, v in test_m.items()))
print(f"{TEST_COUNTRY} only ({aus_mask.sum()} rows): " + ", ".join(f"{k}={v:.4f}" for k, v in aus_m.items()))
if aus_mask.sum() < 10:
    print(f"  (note: only {aus_mask.sum()} {TEST_COUNTRY} test rows, so R2 is very noisy)")


# =====================================================================================
# Part 5 - Saving and visualising the results
# =====================================================================================
section("Part 5 - Save results and charts")
# Test metrics are only filled in for the selected combination, so the test set isn't used for comparison.
is_best = ((results["model"] == best_name) & (results["feature_set"] == best_fs)).to_numpy()

# np.where(condition, if_true, if_false)
# works on every row at once.
# Here it builds a column of 18 values: the test score v on the winning row and a blank (np.nan) on every other row.
for k, v in test_m.items():
    results[f"test_{k}"] = np.where(is_best, v, np.nan)
for k, v in aus_m.items():
    results[f"test_{TEST_COUNTRY.lower()}_{k}"] = np.where(is_best, v, np.nan)
results["test_n_rows"] = np.where(is_best, len(test), np.nan)
results[f"test_{TEST_COUNTRY.lower()}_n_rows"] = np.where(is_best, aus_mask.sum(), np.nan)
results["selected"] = is_best
results.to_csv(os.path.join(OUT_DIR, "world_model_metrics.csv"), index=False)

joblib.dump({"pipeline": best_model, "features": best_features, "target": TARGET},
            os.path.join(OUT_DIR, "world_model.joblib"))

# ---- Step 5b - Chart 1: predicted vs actual
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.4, label="All test rows")
ax.scatter(y_test[aus_mask], y_pred[aus_mask], color="red", label=TEST_COUNTRY, zorder=3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "k--", label="Perfect prediction")
ax.set(xlabel="Actual PM2.5 exposure (µg/m³)", ylabel="Predicted PM2.5 exposure (µg/m³)",
       title=f"World model: predicted vs actual, {SPLIT_YEAR}+\n{best_name}, feature set {best_fs}")
ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "world_model_pred_vs_actual.png"), dpi=150)
plt.close(fig)

# ---- Step 5c - Chart 2: residuals
resid = y_test - y_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred, resid, alpha=0.4)
ax.axhline(0, color="k", linestyle="--")
ax.set(xlabel="Predicted PM2.5 exposure (µg/m³)", ylabel="Residual (actual - predicted)",
       title="World model: residuals on test set")
ax.grid(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "world_model_residuals.png"), dpi=150)
plt.close(fig)

# ---- Step 5d - Chart 3: comparing the models in cross-validation
FS_COLORS = dict(zip(FEATURE_SETS, ["tab:blue", "tab:orange", "tab:green"]))
labels = list(cv_folds)
colors = [FS_COLORS[label.split(" | ")[1]] for label in labels]
fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
for ax, metric in zip(axes, ["R2", "RMSE", "MAE"]):
    means = [cv_folds[l][metric].mean() for l in labels]
    stds = [cv_folds[l][metric].std() for l in labels]
    ax.barh(labels, means, xerr=stds, capsize=3, color=colors)
    ax.set_title(f"CV {metric} (mean ± std)")
    ax.grid(True, axis="x")
axes[0].invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "world_model_cv_comparison.png"), dpi=150)
plt.close(fig)

print(f"Saved metrics, model and plots to {OUT_DIR}/")


# =====================================================================================
# Part 6 - Which groups of features actually help? (ablation, change 3)
# =====================================================================================
section("Part 6 - Ablation: remove one group of features at a time")
ABLATION_MODEL = models["Ridge(alpha=100)"]   # the same pipeline as Step 3a
FEATURES_D = FEATURE_SETS["D (C + power-plant emissions)"]


def ablation_table(variants, reference):
    """CV R² and RMSE for each {name: feature list}, compared with the `reference` feature list."""
    def fold_scores(features):
        res = cross_validate(clone(ABLATION_MODEL), train[features], y_train, groups=groups_train,
                             cv=cv, scoring=scoring)
        return res["test_r2"], -res["test_rmse"]

    ref_r2, _ = fold_scores(reference)
    rows = []
    for name, features in variants.items():
        r2, rmse = fold_scores(features)
        rows.append({
            "variant": name,
            "n_features": len(features),
            "cv_r2_mean": r2.mean(), "cv_r2_std": r2.std(),
            "cv_rmse_mean": rmse.mean(),
            "change_vs_D": r2.mean() - ref_r2.mean(),
            "folds_worse_than_D": int((r2 < ref_r2).sum()),
            "fold_r2": [round(v, 2) for v in r2],
        })
    return pd.DataFrame(rows)


def features_without(features, *groups):
    """The feature list with every feature of the given groups removed."""
    removed = []
    for group in groups:
        removed.extend(group)
    kept = []
    for feature in features:
        if feature not in removed:
            kept.append(feature)
    return kept


GROUPS = {
    "source shares": SOURCE_SHARES,
    "per-person amounts": PER_PERSON,
    "power-plant SO2": ["power_so2_per_kwh", "power_so2_per_capita"],
    "power-plant NOx": ["power_nox_per_kwh", "power_nox_per_capita"],
    "power-plant PM2.5": ["power_pm25_per_kwh", "power_pm25_per_capita"],
}

variants = {"D (all groups)": FEATURES_D}
for group_name, group_features in GROUPS.items():
    variants[f"D minus {group_name}"] = features_without(FEATURES_D, group_features)
variants["D minus all 3 power-plant groups (= C)"] = features_without(
    FEATURES_D, GROUPS["power-plant SO2"], GROUPS["power-plant NOx"], GROUPS["power-plant PM2.5"])

ablation_main = ablation_table(variants, reference=FEATURES_D)
print(ablation_main.round(3).to_string(index=False))

# ---- Step 6b - Is the per-person group just a stand-in for wealth?
TOTAL_PER_PERSON = ["per_capita_electricity"]                                                # (a)
FUEL_PER_PERSON = ["coal_elec_per_capita", "gas_elec_per_capita",
                   "oil_elec_per_capita", "fossil_elec_per_capita"]                          # (b)

split_variants = {
    "D (all groups)": FEATURES_D,
    "(a) total electricity per person, ON ITS OWN": TOTAL_PER_PERSON,
    "(b) fuel-specific per person, ON ITS OWN": FUEL_PER_PERSON,
    "(a) + (b) together, ON ITS OWN": TOTAL_PER_PERSON + FUEL_PER_PERSON,
    "D minus (a) total electricity per person": features_without(FEATURES_D, TOTAL_PER_PERSON),
    "D minus (b) fuel-specific per person": features_without(FEATURES_D, FUEL_PER_PERSON),
    "D minus (a) and (b)": features_without(FEATURES_D, TOTAL_PER_PERSON, FUEL_PER_PERSON),
}
ablation_split = ablation_table(split_variants, reference=FEATURES_D)
print("\nSplit of the per-person group:")
print(ablation_split.round(3).to_string(index=False))

print("\nCorrelation with PM2.5 exposure (training years):")
for column in TOTAL_PER_PERSON + ["fossil_elec_per_capita"]:
    print(f"  {column:26s} {train[column].corr(train[TARGET]):+.2f}")
print(f"Correlation between total and fossil electricity per person: "
      f"{train['per_capita_electricity'].corr(train['fossil_elec_per_capita']):+.2f}")

# Save the numbers and a chart of both tables
ablation_main["table"] = "remove one group from D"
ablation_split["table"] = "split of the per-person group"
pd.concat([ablation_main, ablation_split]).to_csv(os.path.join(OUT_DIR, "ablation_metrics.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
for ax, table, title in [(axes[0], ablation_main, "Remove one group from D"),
                         (axes[1], ablation_split, "Split of the per-person group")]:
    ax.barh(table["variant"], table["cv_r2_mean"], xerr=table["cv_r2_std"], capsize=3, color="tab:green")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.invert_yaxis(); ax.grid(True, axis="x")
    ax.set_title(f"{title}: CV R² (mean ± std)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "ablation.png"), dpi=150)
plt.close(fig)


# =====================================================================================
# Part 7 - A second question: does a country's air change when ITS electricity changes? (change 4)
# =====================================================================================
section("Part 7 - Within-country analysis (each country compared with itself)")

# ---- Step 7a - Express everything relative to each country's own average
WITHIN_FEATURES = FEATURE_SETS["D (C + power-plant emissions)"]
within_cols = WITHIN_FEATURES + ["year", TARGET]

# each country's own average over the TRAINING years only (no test-period information)
country_means = train.groupby("country")[within_cols].mean()


def relative_to_own_average(frame):
    """Each value minus that country's training-period average."""
    means = country_means.reindex(frame["country"]).set_index(frame.index)
    out = frame[within_cols] - means[within_cols]
    out["country"] = frame["country"]
    return out


train_w = relative_to_own_average(train)
test_w = relative_to_own_average(test)
print(f"Typical size of a country's year-to-year swings in PM2.5 (std of deviations): "
      f"{train_w[TARGET].std():.2f} µg/m³ (training years)")

# ---- Step 7b - Compare three sets of inputs with cross-validation
WITHIN_SETS = {
    "Time trend only": ["year"],
    "Electricity only": WITHIN_FEATURES,
    "Electricity + time trend": WITHIN_FEATURES + ["year"],
}
within_models = {name: models[name] for name in ["Ridge(alpha=10)", "RandomForest(depth=5)"]}

within_results, within_folds = cross_validate_all(train_w, TARGET, WITHIN_SETS, within_models)
print(within_results.round(3).to_string(index=False))

# ---- Step 7c - Choose the best and test it once
best_w = within_results.loc[within_results["cv_rmse_mean"].idxmin()]
best_w_name, best_w_set = best_w["model"], best_w["feature_set"]
best_w_features = WITHIN_SETS[best_w_set]
within_model = clone(within_models[best_w_name]).fit(train_w[best_w_features], train_w[TARGET])
y_pred_w = within_model.predict(test_w[best_w_features])
test_w_m = metrics(test_w[TARGET], y_pred_w)

trend = within_results[within_results["feature_set"] == "Time trend only"]["cv_r2_mean"].max()
both = within_results[within_results["feature_set"] == "Electricity + time trend"]["cv_r2_mean"].max()
print(f"\nBest by CV RMSE: {best_w_name} with '{best_w_set}'")
print(f"Test set (2018+): " + ", ".join(f"{k}={v:.4f}" for k, v in test_w_m.items()))
print(f"\nBest CV R2 with time trend only:          {trend:.3f}")
print(f"Best CV R2 with electricity + time trend:  {both:.3f}  (extra explained by electricity: {both - trend:+.3f})")

is_best_w = ((within_results["model"] == best_w_name) & (within_results["feature_set"] == best_w_set)).to_numpy()
for k, v in test_w_m.items():
    within_results[f"test_{k}"] = np.where(is_best_w, v, np.nan)
within_results["selected"] = is_best_w
within_results.to_csv(os.path.join(OUT_DIR, "within_country_metrics.csv"), index=False)

# ---- Step 7d - Charts
SET_COLORS = {"Time trend only": "tab:grey", "Electricity only": "tab:orange",
              "Electricity + time trend": "tab:green"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1.4]})

labels_w = list(within_folds)
ax1.barh(labels_w, [within_folds[l]["R2"].mean() for l in labels_w],
         xerr=[within_folds[l]["R2"].std() for l in labels_w], capsize=3,
         color=[SET_COLORS[l.split(" | ")[1]] for l in labels_w])
ax1.invert_yaxis(); ax1.grid(True, axis="x")
ax1.set_title("Within-country CV R² (mean ± std)")

for country, color in zip(["China", "United States", TEST_COUNTRY], ["tab:red", "tab:blue", "tab:green"]):
    rows = df[df["country"] == country].sort_values("year")
    estimate = within_model.predict(relative_to_own_average(rows)[best_w_features]) + country_means.loc[country, TARGET]
    ax2.plot(rows["year"], rows[TARGET], color=color, label=f"{country} actual")
    ax2.plot(rows["year"], estimate, color=color, linestyle="--", label=f"{country} model")
ax2.axvspan(SPLIT_YEAR - 0.5, df["year"].max() + 0.5, color="grey", alpha=0.15, label="Test period")
ax2.set(xlabel="Year", ylabel="PM2.5 exposure (µg/m³)",
        title=f"Within-country model ({best_w_name}, {best_w_set})")
ax2.legend(fontsize=8, ncol=2); ax2.grid(True)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "within_country.png"), dpi=150)
plt.close(fig)

print(f"\nDone. All results are in {OUT_DIR}/")
