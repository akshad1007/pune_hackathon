# scripts/update_aqi_data.py (Final Simplified Version)

import pandas as pd
from pathlib import Path

print("Starting script to add clean 2023 data to the main AQI file...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
# The new file should be in the 'raw/cpcb_aqi' folder for consistency
NEW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cpcb_aqi" / "AQI_data_2023 - Sheet1.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# --- 2. Load the Datasets ---
try:
    df_main = pd.read_csv(PROCESSED_DATA_PATH / "combined_aqi_data.csv")
    # Load the new, clean 2023 data file you provided
    df_2023_clean = pd.read_csv(NEW_DATA_PATH)
    print("Successfully loaded existing AQI data and new 2023 data.")
except Exception as e:
    print(f"❌ Error: A required file was not found. Details: {e}")
    exit()

# --- 3. Prepare the 2023 Data for Merging ---
print("Preparing the new 2023 data...")

# Rename columns to match your main 'combined_aqi_data.csv' file
rename_map = {
    'State / Union Territory': 'state',
    'City / town': 'city',
    # We will use the PM2.5 average as the primary AQI value for 2023
    'PM2.5_Annual_Average': 'aqi' 
}
df_2023_clean.rename(columns=rename_map, inplace=True)

# Add the 'date' column for the year 2023
df_2023_clean['date'] = '2023-01-01'

# --- 4. Append the Clean 2023 Data to the Main File ---
print("Appending new data to the main AQI file...")
df_updated = pd.concat([df_main, df_2023_clean])

# Clean up and sort the final combined file
df_updated['date'] = pd.to_datetime(df_updated['date'], errors='coerce')
df_updated.sort_values(by=['city', 'date'], inplace=True)
df_updated.drop_duplicates(subset=['city', 'date'], keep='last', inplace=True)

# --- 5. Save the Updated Master AQI File ---
output_filename = PROCESSED_DATA_PATH / "combined_aqi_data.csv"
df_updated.to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS! Your main AQI dataset has been updated with 2023 data.")
print(f"The updated file is saved at: {output_filename}")