 
# scripts/process_aqi_2023.py

import camelot
import pandas as pd
from pathlib import Path

print("Starting script to process 2023 AQI PDF...")

# --- 1. Define Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cpcb_aqi"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

PDF_FILE = RAW_DATA_PATH / "AQI_data_2023.pdf"

# --- 2. Read All Tables from the PDF ---
if not PDF_FILE.exists():
    print(f"❌ Error: PDF file not found at {PDF_FILE}")
else:
    print(f"Reading all tables from {PDF_FILE.name}...")
    
    # 'pages="all"' tells camelot to read the entire document
    tables = camelot.read_pdf(str(PDF_FILE), pages='all', flavor='stream')
    
    print(f"Found {len(tables)} tables in the document.")
    
    if tables:
        # Combine all the tables found into one single DataFrame
        df_2023 = pd.concat([table.df for table in tables], ignore_index=True)
        
        # Save the raw extracted data
        output_filename = PROCESSED_DATA_PATH / "raw_aqi_data_2023.csv"
        df_2023.to_csv(output_filename, index=False)
        
        print(f"\n✅ Success! Raw 2023 AQI data extracted.")
        print(f"Data saved to: {output_filename}")
        print("\nNOTE: You will need to add this new data to your 'combined_aqi_data.csv' file.")
    else:
        print("No tables could be extracted from the PDF.")