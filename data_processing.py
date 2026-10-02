import pandas as pd
import kagglehub
import os

# 1. Download data automatically via kagglehub
path = kagglehub.dataset_download("pralabhpoudel/world-energy-consumption")

# The downloaded file is usually named "World Energy Consumption.csv"
energy_file = os.path.join(path, "World Energy Consumption.csv") 

# 2. Define PM2.5 file path (ensure this file is in your working directory)
pm25_file = 'average-exposure-pm25-pollution.csv'

# 3. Define the cleaning and merging function
def clean_and_merge_climate_data(energy_path, pm25_path, min_valid_years=15):
    energy_df = pd.read_csv(energy_path)
    pm25_df = pd.read_csv(pm25_path)
    
    emitting_cols = [c for c in ['coal_share_elec', 'gas_share_elec', 'oil_share_elec'] if c in energy_df.columns]
    clean_cols = [c for c in ['solar_share_elec', 'wind_share_elec', 'hydro_share_elec', 
                              'nuclear_share_elec', 'other_renewables_share_elec'] if c in energy_df.columns]
    
    # Defragment DataFrame to avoid PerformanceWarning
    energy_df = energy_df.copy()
    
    # Calculate total percentage of emitting and non-emitting sources
    energy_df['% GHG emitting source'] = energy_df[emitting_cols].sum(axis=1, min_count=1)
    energy_df['% GHG non emitting source'] = energy_df[clean_cols].sum(axis=1, min_count=1)
    
    # Extract only necessary columns
    energy_subset = energy_df[['country', 'year', '% GHG emitting source', '% GHG non emitting source']]
    
    # Rename columns in PM2.5 dataset to match for merging
    pm25_df = pm25_df.rename(columns={
        'Entity': 'country',
        'Year': 'year',
        'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)': 'PM2.5 Exposure'
    })
    
    # Merge datasets and drop missing values
    merged_df = pd.merge(energy_subset, pm25_df[['country', 'year', 'PM2.5 Exposure']], on=['country', 'year'], how='inner')
    clean_merged = merged_df.dropna(subset=['% GHG emitting source', '% GHG non emitting source', 'PM2.5 Exposure'])
    
    # Exclude countries with insufficient data (less than min_valid_years)
    valid_years_per_country = clean_merged.groupby('country').size()
    countries_to_keep = valid_years_per_country[valid_years_per_country >= min_valid_years].index
    
    final_df = clean_merged[clean_merged['country'].isin(countries_to_keep)]
    final_df.to_csv('cleaned_climate_merged.csv', index=False)
    
    print(f"Data cleaned successfully! Retained {len(countries_to_keep)} countries/regions.")
    return final_df

# 4. Execute the function
if __name__ == "__main__":
    final_data = clean_and_merge_climate_data(energy_file, pm25_file)