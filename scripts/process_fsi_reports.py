# scripts/process_fsi_reports.py

import camelot
import pandas as pd
from pathlib import Path

print("Starting FSI report processing script...")

PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "raw" / "fsi_reports" / "IFSR2023.pdf"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_PATH.mkdir(exist_ok=True)

if not PDF_PATH.exists():
    print(f"Error: PDF file not found at {PDF_PATH}")
else:
    print(f"Reading tables from {PDF_PATH.name}...")
    try:
        tables = camelot.read_pdf(
            str(PDF_PATH), 
            pages='39-179',  # Correct page range for district tables
            flavor='stream'
        )

        print(f"Found {len(tables)} tables.")

        if tables:
            all_tables_df = pd.concat([table.df for table in tables], ignore_index=True)
            print("Successfully combined all tables.")

            output_filename = PROCESSED_DATA_PATH / "raw_fsi_forest_cover.csv"
            all_tables_df.to_csv(output_filename, index=False)

            print(f"\nSuccess! Raw data extracted.")
            print(f"Data saved to: {output_filename}")
            print("\nNOTE: This raw data will need further cleaning.")

    except Exception as e:
        print(f"An error occurred: {e}") 
