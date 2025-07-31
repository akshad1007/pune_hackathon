# scripts/final_merge.py

import pandas as pd
from pathlib import Path

print("Starting the final merge script...")

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

# --- 3. Prepare Datasets for Merging ---
print("Preparing datasets for merging...")

# AQI Data: Calculate annual average AQI for 2020
df_aqi['date'] = pd.to_datetime(df_aqi['date'])
df_aqi_2020 = df_aqi[df_aqi['date'].dt.year == 2020].copy()
df_aqi_agg = df_aqi_2020.groupby('city')['aqi'].mean().reset_index()
df_aqi_agg.rename(columns={'city': 'DistrictName', 'aqi': 'AvgAQI_2020'}, inplace=True)

# Forest Data: Ensure column names are consistent
df_forest.rename(columns={'district': 'DistrictName'}, inplace=True)

# --- 4. Merge Datasets ---
print("Merging datasets...")
df_final = pd.merge(df_master, df_aqi_agg, on='DistrictName', how='left')
df_final = pd.merge(df_final, df_forest, on='DistrictName', how='left')
df_final = pd.merge(df_final, df_vehicles, on=['DistrictName', 'StateName'], how='left')

# --- 5. Final Calculations (Example) ---
print("Performing final calculations...")
if 'total_forest_cover' in df_final.columns and 'DistrictVehicles_2020' in df_final.columns:
    df_final['VehiclesPerCapita'] = df_final['DistrictVehicles_2020'] / df_final['Population']
    df_final['ForestPerCapita_sqkm'] = df_final['total_forest_cover'] / df_final['Population']

# --- 6. Save the Final Master Dataset ---
output_filename = OUTPUT_PATH / "India_District_Environmental_Indicators.csv"
df_final.to_csv(output_filename, index=False)

print("\n\n" + "="*50)
print("🎉 PROJECT COMPLETE! 🎉")
print("="*50)
print(f"\nYour final master dataset has been created and saved to:")
print(output_filename) 
