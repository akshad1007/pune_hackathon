# scripts/process_morth_data.py (Final Working Version)

import pandas as pd
from pathlib import Path

print("Starting MoRTH vehicle data processing script...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# --- 2. Load Data Files ---
try:
    df_districts = pd.read_csv(PROCESSED_DATA_PATH / "District_Master_Key_with_Population.csv")
    df_city_transport = pd.read_csv(RAW_DATA_PATH / "morth_reports" / "morth_cities_transport_2020.csv")
    df_city_non_transport = pd.read_csv(RAW_DATA_PATH / "morth_reports" / "morth_cities_non_transport_2020.csv")
    df_state_commercial = pd.read_csv(RAW_DATA_PATH / "morth_reports" / "morth_state_commercial_2020.csv")
    print("Successfully loaded all data files.")
except Exception as e:
    print(f"❌ Error: A required file was not found. Please check your file locations. Details: {e}")
    exit()

# --- 3. Process City Data ---
print("Processing city-level anchor data...")
df_city_transport.rename(columns={"Million Plus Cities": "MORTH_City_Name", "Transport - Total Transport": "TransportVehicles"}, inplace=True)
df_city_non_transport.rename(columns={"Million Plus Cities": "MORTH_City_Name", "Grand Total (Transport+Non Transport)": "TotalVehicles"}, inplace=True)

df_cities_merged = pd.merge(df_city_transport[['MORTH_City_Name', 'TransportVehicles']], df_city_non_transport[['MORTH_City_Name', 'TotalVehicles']],on="MORTH_City_Name")

df_final = pd.merge(df_districts, df_cities_merged[['MORTH_City_Name', 'TotalVehicles']], on="MORTH_City_Name", how="left")

# --- 4. Disaggregate State Data ---
print("Disaggregating state-level data...")

# --- THIS SECTION IS NOW FIXED ---
# Using the actual column names from your file
df_state_commercial.rename(columns={
    "States/Union Territories": "StateName", 
    "Total Commercial Vehicles in Use": "StateTotalVehicles"
}, inplace=True)
# --------------------------------

state_population_totals = df_districts.groupby('StateName')['Population'].sum().reset_index()
state_anchor_vehicles = df_final.groupby('StateName')['TotalVehicles'].sum().reset_index()
state_anchor_population_df = df_final[df_final['TotalVehicles'].notna()]
state_anchor_population = state_anchor_population_df.groupby('StateName')['Population'].sum().reset_index().rename(columns={'Population': 'AnchorPopulation'})

df_state_stats = pd.merge(state_population_totals, state_anchor_vehicles, on="StateName", how="left").fillna(0)
df_state_stats = pd.merge(df_state_stats, state_anchor_population, on="StateName", how="left").fillna(0)
df_state_stats = pd.merge(df_state_stats, df_state_commercial[['StateName', 'StateTotalVehicles']], on="StateName", how="left").fillna(0)

df_state_stats['RemainingVehicles'] = df_state_stats['StateTotalVehicles'] - df_state_stats['TotalVehicles']
df_state_stats['RemainingPopulation'] = df_state_stats['Population'] - df_state_stats['AnchorPopulation']
df_state_stats.loc[df_state_stats['RemainingVehicles'] < 0, 'RemainingVehicles'] = 0

# --- 5. Apportion Vehicles ---
df_final = pd.merge(df_final, df_state_stats[['StateName', 'RemainingVehicles', 'RemainingPopulation']], on="StateName", how="left")

non_anchor_mask = df_final['TotalVehicles'].isna()
df_final.loc[non_anchor_mask & (df_final['RemainingPopulation'] > 0), 'PopulationShare'] = df_final['Population'] / df_final['RemainingPopulation']
df_final.loc[non_anchor_mask, 'EstimatedVehicles'] = df_final['PopulationShare'] * df_final['RemainingVehicles']

df_final['DistrictVehicles_2020'] = df_final['TotalVehicles'].fillna(df_final['EstimatedVehicles'])

# --- 6.