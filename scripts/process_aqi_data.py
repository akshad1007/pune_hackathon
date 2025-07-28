# scripts/process_aqi_data.py (FINAL VERSION 2.1 - With Duplicate Column Fix)

import pandas as pd
from pathlib import Path
import re

print("Starting AQI data processing...")

# Define project paths
PROJECT_ROOT = Path(__file__).parent.parent 
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cpcb_aqi"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH.mkdir(exist_ok=True)

all_aqi_files = list(RAW_DATA_PATH.glob("*.csv"))

if not all_aqi_files:
    print(f"Error: No CSV files found in {RAW_DATA_PATH}")
else:
    print(f"Found {len(all_aqi_files)} AQI files to process.")

    df_list = []
    for f in all_aqi_files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
            df_list.append(df)
            print(f"Successfully read {f.name}")
        except Exception as e:
            print(f"Could not read file {f.name}. Error: {e}")

    if df_list:
        print("Combining all data...")
        full_aqi_df = pd.concat(df_list, ignore_index=True)

        print("Harmonizing column names...")

        original_columns = full_aqi_df.columns
        new_columns = []
        for col in original_columns:
            new_col = str(col).lower()
            new_col = re.sub(r'[^a-zA-Z0-9_]', '_', new_col)
            new_columns.append(new_col)
        full_aqi_df.columns = new_columns
        
        # --- NEW LINE ADDED HERE ---
        # 4.1a: Remove duplicate columns (e.g., if 'City' and 'city' both became 'city')
        # This keeps the first occurrence of each column name.
        full_aqi_df = full_aqi_df.loc[:, ~full_aqi_df.columns.duplicated()]

        merge_map = {
            'city': ['city__town__district', 'city_town'],
            'aqi': ['index_value'],
            'aqi_category': ['air_quality'],
            'no_stations': ['no__stations'],
            'prominent_pollutant': ['prominent_pollutant']
        }
        
        for final_col, old_cols in merge_map.items():
            for old_col in old_cols:
                if old_col in full_aqi_df.columns and final_col in full_aqi_df.columns:
                    full_aqi_df[final_col] = full_aqi_df[final_col].fillna(full_aqi_df[old_col])
        
        final_columns_to_keep = [
            'date', 'city', 'no_stations', 'aqi', 'aqi_category', 
            'prominent_pollutant', 'pm2_5', 'pm10', 'no', 'no2', 'nox', 
            'nh3', 'co', 'so2', 'o3'
        ]
        
        existing_final_columns = [col for col in final_columns_to_keep if col in full_aqi_df.columns]
        full_aqi_df = full_aqi_df[existing_final_columns]

        print("Cleaning data...")
        if 'date' in full_aqi_df.columns:
            full_aqi_df['date'] = pd.to_datetime(full_aqi_df['date'], errors='coerce')
        
        for col in full_aqi_df.columns:
            if col not in ['date', 'city', 'aqi_category', 'prominent_pollutant']:
                full_aqi_df[col] = pd.to_numeric(full_aqi_df[col], errors='coerce')

        full_aqi_df.dropna(subset=['date'], inplace=True)
        full_aqi_df.sort_values(by=['city', 'date'], inplace=True)
        
        output_filename = PROCESSED_DATA_PATH / "combined_aqi_data.csv"
        full_aqi_df.to_csv(output_filename, index=False)

        print(f"\n✅ SUCCESS! Project complete.")
        print(f"Combined and cleaned data saved to: {output_filename}")
    else:
        print("\n❌ Error: No files were successfully read.")