# scripts/final_merge.py (Final Robust Version)

import pandas as pd
from pathlib import Path

print("Starting the final, robust merge script...")

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

# Function to create a clean, uppercase key for merging
def create_merge_key(df, col_name):
    # Ensure the column exists before trying to access .str
    if col_name in df.columns:
        df['merge_key'] = df[col_name].astype(str).str.upper().str.strip()
    return df

df_master = create_merge_key(df_master, 'DistrictName')
df_aqi = create_merge_key(df_aqi.rename(columns={'city': 'DistrictName'}), 'DistrictName')
df_forest = create_merge_key(df_forest.rename(columns={'district': 'DistrictName'}), 'DistrictName')
df_vehicles = create_merge_key(df_vehicles, 'DistrictName')

# --- 4. Prepare Datasets for Merging ---
df_aqi['date'] = pd.to_datetime(df_aqi['date'], errors='coerce')
df_aqi_2020 = df_aqi[df_aqi['date'].dt.year == 2020].copy()
df_aqi_agg = df_aqi_2020.groupby('merge_key')['aqi'].mean().reset_index()
df_aqi_agg.rename(columns={'aqi': 'AvgAQI_2020'}, inplace=True)

# --- 5. Merge Datasets using the Standardized Key ---
print("Merging datasets...")
df_final = pd.merge(df_master, df_aqi_agg, on='merge_key', how='left')
df_final = pd.merge(df_final, df_forest, on='merge_key', how='left', suffixes=('', '_forest'))
df_final = pd.merge(df_final, df_vehicles, on='merge_key', how='left', suffixes=('', '_vehicles'))

# --- 6. Clean Up and Finalize ---
# This is the corrected part. We rename the columns we know and keep.
df_final.rename(columns={
    'total_forest_cover': 'TotalForestCover_sqkm',
    'DistrictVehicles_2020': 'EstimatedVehicles_2020'
}, inplace=True)

# --- 7. Final Imputation and Saving ---
print("Filling any remaining missing values and saving...")
cols_to_fill = ['AvgAQI_2020', 'TotalForestCover_sqkm', 'EstimatedVehicles_2020', 'Population']
for col in cols_to_fill:
    if col in df_final.columns:
        df_final[col] = df_final.groupby('StateName')[col].transform(lambda x: x.fillna(x.mean()))
df_final.fillna(0, inplace=True)

output_filename = OUTPUT_PATH / "India_District_Environmental_Indicators.csv"
df_final.to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS! Your final master dataset has been created.")
print(f"File saved to: {output_filename}")