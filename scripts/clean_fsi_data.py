# scripts/clean_fsi_data.py (FINAL VERSION 3.0)

import pandas as pd
from pathlib import Path
import re

print("Starting FSI data cleaning script (v3.0)...")

# 1. Define Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "raw_fsi_forest_cover.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

try:
    # 2. Load Raw Data without Headers
    # We ignore the existing messy headers completely.
    df = pd.read_csv(RAW_DATA_PATH, header=None)
    print("Successfully loaded raw FSI data.")
except FileNotFoundError:
    print(f"❌ Error: Raw file not found at {RAW_DATA_PATH}")
    exit()

# 3. Find Rows Containing District Data
# We'll identify data rows by looking for a specific, common value like "Grand Total" to know where a table ends.
# A more robust method is to find rows that are likely to be data.
# Let's assume a valid data row starts with a district name (text) and has numbers later.
# For simplicity, we'll find the first valid header and work from there.

header_row_index = df[df.iloc[:, 0].str.contains('District', na=False)].index[0]
df_data = df.iloc[header_row_index + 1:].copy()

# 4. Select Columns by Position and Rename Manually
# Based on the PDF, the columns are in this order:
# 0=District, 1=GeoArea, 2=VDF, 3=MDF, 4=OF, 5=Total, 6=%, 7=Change, 8=Scrub
try:
    df_clean = df_data[[0, 2, 3, 4, 5, 8]].copy()
    df_clean.columns = [
        'district', 
        'very_dense_forest', 
        'mod_dense_forest', 
        'open_forest', 
        'total_forest_cover', 
        'scrub'
    ]
    print("Manually selected and renamed data columns.")
except Exception as e:
    print(f"Error selecting columns. The table structure may be unexpected. Error: {e}")
    exit()

# 5. Remove non-data rows (like "Grand Total")
df_clean = df_clean[~df_clean['district'].str.contains('Grand Total', na=False)]
print(f"Filtered out summary rows. Found {len(df_clean)} district rows.")

# 6. Clean and Convert Data Types
for col in ['very_dense_forest', 'mod_dense_forest', 'open_forest', 'total_forest_cover', 'scrub']:
    # Remove commas and convert to numbers. 'coerce' turns errors into NaN (Not a Number).
    df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')

# Drop any rows that failed to convert properly
df_clean.dropna(inplace=True)
print("Cleaned and converted data to numeric types.")

# 7. Save the Final Clean File
output_filename = PROCESSED_DATA_PATH / "cleaned_fsi_forest_cover.csv"
df_clean.to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS! Cleaned forest cover data is complete.")
print(f"Data saved to: {output_filename}")