# scripts/add_city_names.py (Version 2.0 - With Advanced Cleaning)

import pandas as pd
from pathlib import Path

print("Starting script to automatically add MoRTH city names to the master file...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

MASTER_FILE_PATH = PROCESSED_DATA_PATH / "District_Master_Key_with_Population.csv"

# --- 2. Load Files ---
try:
    df_master = pd.read_csv(MASTER_FILE_PATH)
    # Reading from the clean CSV file you created
    df_city_list = pd.read_csv(RAW_DATA_PATH / "morth_reports" / "morth_cities_non_transport_2020.csv")
    print("Successfully loaded master district file and MoRTH city list.")
except Exception as e:
    print(f"❌ Error: Could not load a file. Details: {e}")
    exit()

# --- 3. Create Mapping and Update Master File ---
# UPDATED: Expanded dictionary to handle more complex name mismatches
city_district_mapping = {
    'Bengaluru': 'Bengaluru Urban',
    'Delhi': 'New Delhi',
    'Durg Bhilai': 'Durg',
    'Greater Mumbai': 'Mumbai',
    'Jamshedpur': 'East Singhbhum',
    'Kalyan Dombivali': 'Thane',
    'Kanpur': 'Kanpur Nagar',
    'Kozhikoda': 'Kozhikode',
    'Pimprichichwad': 'Pune',
    'Trichy': 'Tiruchirappalli',
    'Vashi N.Mumbai': 'Thane',
    'Vasai': 'Palghar'
}

# Clean up column name and city names
df_city_list.rename(columns={"Million Plus Cities": "CityName"}, inplace=True)
# UPDATED: Strip whitespace from city names and filter out the 'Total' row
df_city_list['CityName'] = df_city_list['CityName'].str.strip()
df_city_list = df_city_list[df_city_list['CityName'] != 'Total'].copy()

morth_cities = df_city_list['CityName'].dropna().unique()

print(f"Found {len(morth_cities)} major cities in the MoRTH data to map.")

for city in morth_cities:
    # Find the official district name, using the map for special cases
    district_name = city_district_mapping.get(city, city)
    
    # Find the row in the master file that matches the district name
    mask = df_master['DistrictName'] == district_name
    
    if mask.any():
        df_master.loc[mask, 'MORTH_City_Name'] = city
    else:
        print(f"  - Warning: Could not find a matching district for the city '{city}' (District: '{district_name}')")

# --- 4. Save the Updated Master File ---
df_master.to_csv(MASTER_FILE_PATH, index=False)

print(f"\n✅ SUCCESS! The master file has been updated with MoRTH city names.")
print(f"File saved to: {MASTER_FILE_PATH}")