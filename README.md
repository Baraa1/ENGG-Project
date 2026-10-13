# ENGG-Project

Air Pollution prediction model

**Question:** can we predict a country's air pollution (PM2.5 exposure) from how it generates its electricity?

[Datasets](https://unisydneyedu-my.sharepoint.com/:f:/r/personal/bali0372_uni_sydney_edu_au/Documents/Semester-5/ENGG2112/Assignments/Project-Datasets?d=wa9a242eb4bf241659d310d7e3871f9f3&csf=1&web=1&e=Xo1ukb)

## Repository structure

```text
ENGG-Project/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md                          sources, citations, what each file contains
│   ├── raw/                               original downloads, never edited
│   │   ├── average-exposure-pm25-pollution.csv
│   │   ├── EDGAR_SO2_1970_2022.xlsx
│   │   ├── EDGAR_NOx_1970_2022.xlsx
│   │   └── EDGAR_PM2.5_1970_2022.xlsx
│   └── processed/
│       └── cleaned_climate_merged.csv     made by notebook 01 / data_processing.py
├── notebooks/                             step-by-step, with plain-English explanations
│   ├── 01_data_cleaning_and_baseline.ipynb
│   ├── 02_world_model.ipynb
│   └── 03_europe_model.ipynb
├── scripts/                               plain Python, run from the command line
│   ├── data_processing.py                 builds data/processed/cleaned_climate_merged.csv
│   ├── ml_model.py                        first Random Forest (random split)
│   ├── world_model.py                     same as notebook 02
│   └── europe_model.py                    same as notebook 03 (Europe regional model)
└── results/                               made by notebook 02 / world_model.py (metrics, model, charts)
    └── europe/                            made by notebook 03 / europe_model.py
```

| Notebook | What it does |
| --- | --- |
| `01_data_cleaning_and_baseline.ipynb` | Downloads the energy data, merges it with PM2.5 exposure, cleans it and saves `data/processed/cleaned_climate_merged.csv`. Also trains a first Random Forest. Note: that first model is tested on a *random* split, so its scores are optimistic; use notebook 02 for the benchmark. |
| `02_world_model.ipynb` | The **world model**: the benchmark that the regional models (Americas, Europe, Asia, Australia) are compared against. Trained on all countries, tested on 2018-2022. Part 6 shows which inputs actually help (an ablation); Part 7 is a separate, second question that compares each country with itself over time. |
| `03_europe_model.ipynb` | The **Europe regional model** (Nhi): 41 European countries, trained on 2000-2018, tested on 2019-2022, using emitting share, year and country. Includes baselines and a control without the emitting share. Its set-up differs from the world model's (see its Summary), so the two scores can't be compared directly yet. |

All notebooks are written for readers without a software background, with an explanation before every step.

| Script | What it does |
| --- | --- |
| `data_processing.py` | Same cleaning as notebook 01: builds `data/processed/cleaned_climate_merged.csv`. |
| `ml_model.py` | The original first Random Forest on a *random* split (optimistic scores). Notebook 01 has since added `max_depth=10` and an F1 score, so their numbers differ. |
| `world_model.py` | Same steps, settings and outputs as notebook 02, without the explanations. Use either one. |
| `europe_model.py` | Same steps, settings and outputs as notebook 03 (Europe regional model, Nhi). Writes to `results/europe/`. |

The scripts work from any folder, e.g. `python scripts/world_model.py` from the repo root.

## Setup

1. Create and activate a virtual environment, then install the packages:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```

2. Either open the notebooks in VS Code or Jupyter (`pip install jupyter`, then `jupyter notebook`) and use **Run All**,
   or run the scripts, e.g. `python scripts/world_model.py`.
   Run notebook 01 / `data_processing.py` only if you need to rebuild the cleaned data; it is already in `data/processed/`.

The World Energy Consumption dataset is downloaded automatically from Kaggle by `kagglehub` the first time a notebook or script runs; it is not stored in the repo.

## Current world model results

Benchmark as of 7 October 2026 (rerun notebook 02 to update). Trained on 1990-2017, tested once on 2018-2022, 190 countries.

| | Value |
| --- | --- |
| Selected model | Ridge regression (alpha = 100) with feature set D (electricity mix, per-person generation and power-plant SO₂ / NOx / PM2.5 emissions) |
| Cross-validation R² (5-fold, grouped by country) | 0.209 |
| Test R² / RMSE / MAE, all countries | 0.344 / 12.66 / 9.53 µg/m³ |
| Test RMSE / MAE, Australia only (5 rows, treat as anecdotal) | 3.95 / 3.57 µg/m³ |

Full tables are in `results/world_model_metrics.csv` and `results/within_country_metrics.csv`.

**How to read this:** power plants produce only about 5% of directly emitted PM2.5 worldwide (most comes from household fuel burning, industry, fires and natural dust), so electricity generation can only ever explain part of the differences in air pollution between countries. Comparing each country with itself over time (notebook 02, Part 7), changes in electricity generation explain only slightly more than a simple time trend.

**Which inputs matter (notebook 02, Part 6, ablation):** the model's skill comes almost entirely from the per-person fossil-fuel amounts (coal, gas, oil and fossil electricity per person). Removing them drops cross-validation R² from 0.21 to about 0. Total electricity per person (a wealth proxy) adds little, the individual source shares add nothing measurable, and the power-plant SO₂ / NOx / PM2.5 emissions add only about +0.03 R², within the fold-to-fold noise. Differences smaller than about 0.1 R² should not be over-interpreted: the spread between folds is about 0.2.

**For regional models:** use the same features, the same 2018 split and the same metrics so the comparison with this benchmark is fair. The saved model can be loaded with:

```python
import joblib
bundle = joblib.load("results/world_model.joblib")
predictions = bundle["pipeline"].predict(new_data[bundle["features"]])
```

## Git Workflow

We keep `main` always working. All changes happen on a separate branch, then get merged in through a Pull Request (PR).

1. **Get the latest changes**

   ```bash
   git checkout main
   git pull
   ```

2. **Create a new branch for your work**

   ```bash
   git checkout -b your-name/short-description
   ```

   Example: `git checkout -b baraa/fix-data-cleaning`

3. **Make your changes, then commit them**

   ```bash
   git add .
   git commit -m "Describe what you changed"
   ```

4. **Push your branch to GitHub**

   ```bash
   git push -u origin your-name/short-description
   ```

5. **Open a Pull Request**
   - Go to the repository on GitHub.
   - You'll see a prompt to open a PR for your branch — click it (or open one manually from the "Pull requests" tab).
   - Add a short description of what you changed, then click "Create pull request".

6. **Merge**
   - Once someone reviews and approves the PR (or if working solo, once you've double-checked it), click "Merge pull request" on GitHub.
   - Delete the branch after merging (GitHub will offer a button for this).

7. **Update your local `main`**

   ```bash
   git checkout main
   git pull
   ```

**Rule of thumb:** never commit directly to `main` — always branch, push, and PR.
