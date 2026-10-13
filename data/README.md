# Data

Files in `raw/` are original downloads and should never be edited by hand. Anything derived from them goes in `processed/` and is produced by a notebook.

## raw/

### `average-exposure-pm25-pollution.csv` - the target variable

- **What:** mean annual exposure to PM2.5 air pollution per country and year, in micrograms per cubic metre (µg/m³).
- **Columns:** `Entity` (country), `Code` (ISO 3-letter code), `Year`, `PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)`.
- **Source:** Our World in Data, from the World Bank World Development Indicators.

### `EDGAR_SO2_1970_2022.xlsx`, `EDGAR_NOx_1970_2022.xlsx`, `EDGAR_PM2.5_1970_2022.xlsx` - power-plant emissions (feature set D)

- **What:** yearly emissions of sulphur dioxide (SO₂), nitrogen oxides (NOx) and fine particles (PM2.5) per country, broken down by sector, 1970-2022, in Gg (thousand tonnes).
- **Used:** sheet `IPCC 2006`, sector code `1.A.1.a` ("Main Activity Electricity and Heat Production"), fossil and bio rows added together. Notebook 02 also uses the PM2.5 file's other sectors to show how small the power sector's share is.
- **Matching:** join on the ISO 3-letter code (`Country_code_A3`), not the country name, because EDGAR spells some names differently.
- **Source:** EDGAR v8.1 air pollutants, <https://edgar.jrc.ec.europa.eu/dataset_ap81> (readme: <https://edgar.jrc.ec.europa.eu/readme/v81_ft2022_ap_readme.txt>).
- **Required citation:** European Commission, Joint Research Centre (EC-JRC). Emissions Database for Global Atmospheric Research (EDGAR), release EDGAR v8.1 (1970 - 2022) of August 2024. Users of the data must acknowledge the source with a reference to the EDGAR v8.1 air pollutant website.

### Not stored here: World Energy Consumption

- **What:** electricity generation by source, per person and in total, plus population, per country and year.
- **Source:** Kaggle dataset `pralabhpoudel/world-energy-consumption` (based on Our World in Data energy data).
- The notebooks download it automatically with `kagglehub`, so it isn't committed (it's also covered by `.gitignore`).

## processed/

### `cleaned_climate_merged.csv`

- **Made by:** `notebooks/01_data_cleaning_and_baseline.ipynb` or `scripts/data_processing.py` (same cleaning, same output).
- **What:** the energy and PM2.5 datasets merged on country and year, with two summary features (`% GHG emitting source`, `% GHG non emitting source`) and the target (`PM2.5 Exposure`). Rows with missing values are dropped, and countries with fewer than 15 years of data are removed.
- **Note:** it still contains 6 aggregate rows (World, European Union (27) and four income groups). Notebook 02 removes them, because they overlap with individual countries.
