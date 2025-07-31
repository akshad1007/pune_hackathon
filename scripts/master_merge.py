# scripts/master_merge.py (Final Master Merging Script)

import pandas as pd
from pathlib import Path

print("Starting the final master merge script...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

# --- 2. Load All Cleaned Datasets ---
try:
    print("Loading datasets...")
    df_master = pd.read_csv(PROCESSED_PATH / "District_Master_Key_with_Population.csv")
    df_aqi = pd.read_csv(PROCESSED_PATH / "combined_aqi_data.csv")
    df_forest = pd.read_csv(PROCESSED_PATH / "cleaned_fsi_forest_cover.csv")
    df_vehicles = pd.read_csv(PROCESSED_PATH / "estimated_district_vehicles_2020.csv")
    print("All datasets loaded successfully.")
except Exception as e:
    print(f"❌ Error loading a file. Details: {e}")
    exit()

# --- 3. STANDARDİZE DISTRICT NAMES ACROSS ALL FILES ---
print("Standardizing district names for a clean merge...")

# Create a clean, uppercase key for merging
df_master['merge_key'] = df_master['DistrictName'].str.upper()

df_aqi.rename(columns={'city': 'DistrictName'}, inplace=True)
df_aqi['merge_key'] = df_aqi['DistrictName'].str.upper()

df_forest.rename(columns={'district': 'DistrictName'}, inplace=True)
df_forest['merge_key'] = df_forest['DistrictName'].str.upper()

df_vehicles['merge_key'] = df_vehicles['DistrictName'].str.upper()

# --- 4. Prepare Datasets for Merging ---
# AQI Data: Calculate annual average AQI for 2020
df_aqi['date'] = pd.to_datetime(df_aqi['date'])
df_aqi_2020 = df_aqi[df_aqi['date'].dt.year == 2020].copy()
df_aqi_agg = df_aqi_2020.groupby('merge_key')['aqi'].mean().reset_index()
df_aqi_agg.rename(columns={'aqi': 'AvgAQI_2020'}, inplace=True)

# --- 5. Merge Datasets using the Standardized Key ---
print("Merging datasets...")
df_final = pd.merge(df_master, df_aqi_agg, on='merge_key', how='left')
df_final = pd.merge(df_final, df_forest, on='merge_key', how='left', suffixes=('', '_forest'))
df_final = pd.merge(df_final, df_vehicles, on='merge_key', how='left', suffixes=('', '_vehicles'))

# --- 6. Clean Up and Finalize ---
# Select and rename final columns
final_cols = {
    'LGD_Code': 'LGD_Code',
    'DistrictName': 'DistrictName',
    'StateName': 'StateName',
    'Population': 'Population',
    'AvgAQI_2020': 'AvgAQI_2020',
    'total_forest_cover': 'TotalForestCover_sqkm',
    'DistrictVehicles_2020': 'EstimatedVehicles_2020'
}
df_final = df_final[list(final_cols.keys())].rename(columns=final_cols)

# --- 7. Final Imputation (Fill remaining NaNs) ---
print("Filling any remaining missing values with state averages...")
cols_to_fill = ['AvgAQI_2020', 'TotalForestCover_sqkm', 'EstimatedVehicles_2020']
for col in cols_to_fill:
    df_final[col] = df_final.groupby('StateName')[col].transform(lambda x: x.fillna(x.mean()))
df_final.fillna(0, inplace=True) # Fill any states that had no data with 0

# --- 8. Save the Final Master Dataset ---
output_filename = OUTPUT_PATH / "India_District_Environmental_Indicators.csv"
df_final.to_csv(output_filename, index=False)

print("\n\n" + "="*50)
print("🎉 PROJECT COMPLETE! 🎉")
print("="*50)
print(f"\nYour final master dataset has been created with far fewer missing values.")
print(f"File saved to: {output_filename}") 
