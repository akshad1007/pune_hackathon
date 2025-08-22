# scripts/process_aqi_data.py (Robust Final Version)

import pandas as pd
from pathlib import Path

print("Starting robust AQI data processing...")

PROJECT_ROOT = Path(__file__).parent.parent 
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cpcb_aqi"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH.mkdir(exist_ok=True)

all_aqi_files = list(RAW_DATA_PATH.glob("*.csv"))
print(f"Found {len(all_aqi_files)} AQI files to process.")

df_list = []
for f in all_aqi_files:
    try:
        # Read the CSV and immediately try to clean it
        df = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
        df_list.append(df)
        print(f"Successfully read {f.name}")
    except Exception as e:
        print(f"Could not read file {f.name}. Error: {e}")

if df_list:
    print("Combining all data...")
    # Use an outer join to handle different column names gracefully
    full_aqi_df = pd.concat(df_list, ignore_index=True, sort=False)

    print("Harmonizing column names...")
    # Create a mapping of all possible column names to a standard set
    column_map = {
        'Date': 'date', 'date': 'date',
        'City': 'city', 'city': 'city',
        'AQI': 'aqi', 'Index Value': 'aqi',
        'Air Quality': 'aqi_category', 'aqi_category': 'aqi_category',
        'Prominent Pollutant': 'prominent_pollutant'
    }

    # Rename columns based on the map
    full_aqi_df.rename(columns=lambda c: column_map.get(c, c), inplace=True)

    # Convert AQI column to numeric, forcing errors into NaN (blanks)
    if 'aqi' in full_aqi_df.columns:
        full_aqi_df['aqi'] = pd.to_numeric(full_aqi_df['aqi'], errors='coerce')

    # Keep only the essential columns
    essential_columns = ['date', 'city', 'aqi', 'aqi_category', 'prominent_pollutant']
    final_df = full_aqi_df[[col for col in essential_columns if col in full_aqi_df.columns]]

    # Remove rows where 'aqi' is completely blank after our cleaning
    final_df.dropna(subset=['aqi'], inplace=True)

    output_filename = PROCESSED_DATA_PATH / "combined_aqi_data.csv"
    final_df.to_csv(output_filename, index=False)

    print(f"\n✅ SUCCESS! A clean version of 'combined_aqi_data.csv' has been created.")
    print(f"Saved to: {output_filename}")
else:
    print("\n❌ Error: No files were successfully read.")