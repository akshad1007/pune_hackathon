# scripts/merge_lgd_census.py (Final Working Version)

import pandas as pd
from pathlib import Path

print("Starting script to merge LGD and Census data...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# --- 2. Load the Source Files ---
try:
    df_lgd = pd.read_csv(PROCESSED_DATA_PATH / "lgd_districts.csv", encoding='utf-8-sig')
    df_census = pd.read_csv(RAW_DATA_PATH / "census_data" / "2011-IndiaStateDistSbDistTwn-0000.csv")
    print("Successfully loaded LGD and Census data files.")
except Exception as e:
    print(f"❌ Error: A required file was not found. Details: {e}")
    exit()

# --- 3. Prepare Both Datasets for Merging ---
print("Preparing data for merge...")

# Prepare the LGD Data
df_lgd.columns = df_lgd.columns.str.strip().str.strip('"')
df_lgd_clean = df_lgd[['state_name_english', 'district_name_english', 'district_code']].copy()
df_lgd_clean.rename(columns={
    'state_name_english': 'StateName',
    'district_name_english': 'DistrictName',
    'district_code': 'LGD_Code'
}, inplace=True)
df_lgd_clean['StateName_upper'] = df_lgd_clean['StateName'].str.upper().str.strip()
df_lgd_clean['DistrictName_upper'] = df_lgd_clean['DistrictName'].str.upper().str.strip()

# Prepare the Census Data
# --- THIS SECTION IS NOW FIXED ---
# Using the exact column names you provided: 'Level', 'TRU', 'State', 'Name', 'TOT_P'
df_census_districts = df_census[(df_census['Level'] == 'DISTRICT') & (df_census['TRU'] == 'Total')].copy()
state_mapping = df_census[df_census['Level'] == 'STATE'][['State', 'Name']].set_index('State')['Name'].to_dict()
df_census_districts['StateName'] = df_census_districts['State'].map(state_mapping)
df_census_clean = df_census_districts[['StateName', 'Name', 'TOT_P']].copy()
df_census_clean.rename(columns={'Name': 'DistrictName', 'TOT_P': 'Population'}, inplace=True)
# --------------------------------

df_census_clean['StateName_upper'] = df_census_clean['StateName'].str.upper().str.strip()
df_census_clean['DistrictName_upper'] = df_census_clean['DistrictName'].str.upper().str.strip()

# --- 4. Merge the Two Datasets ---
print("Merging the two datasets on State and District names...")
df_master = pd.merge(
    df_lgd_clean,
    df_census_clean[['StateName_upper', 'DistrictName_upper', 'Population']],
    on=['StateName_upper', 'DistrictName_upper'],
    how='left'
)

# --- 5. Finalize and Save the Master File ---
df_master['MORTH_City_Name'] = ''
final_columns = ['LGD_Code', 'DistrictName', 'StateName', 'Population', 'MORTH_City_Name']
df_master = df_master[final_columns]

output_filename = PROCESSED_DATA_PATH / "District_Master_Key_with_Population.csv"
df_master.to_csv(output_filename, index=False)

print(f"\n✅ SUCCESS! The master district file with population has been created automatically.")
print(f"File saved to: {output_filename}")
print("\nYour next step is to manually edit this new file to add the city names.")