import shutil
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("data")
OUTPUT_DIR = Path("filtered_minerals")
REVIEW_CSV = "selected.csv"

def main():
    if not Path(REVIEW_CSV).exists():
        print(f"Error: Could not find '{REVIEW_CSV}'. Please run the first script and curate the list.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading curated list from {REVIEW_CSV}...")
    df = pd.read_csv(REVIEW_CSV)
    
    files_copied = 0
    missing_files = 0

    for index, row in df.iterrows():
        mineral = row['Target_Mineral']
        spectrum_id = row['SpectrumID'].lower()
        sample_id = row['SampleID'].lower()
        pi_code = row['PI_Code'].lower()
        
        first_two = sample_id[:2]
        
        # Try .txt first, fallback to .asc
        source_txt = DATA_DIR / pi_code / first_two / f"{spectrum_id}.txt"
        source_asc = DATA_DIR / pi_code / first_two / f"{spectrum_id}.asc"
        
        target_name = f"{mineral}_{spectrum_id.upper()}"
        
        if source_txt.exists():
            shutil.copy2(source_txt, OUTPUT_DIR / f"{target_name}.txt")
            files_copied += 1
        elif source_asc.exists():
            shutil.copy2(source_asc, OUTPUT_DIR / f"{target_name}.asc")
            files_copied += 1
        else:
            print(f"Warning: Could not find data file for {spectrum_id.upper()}")
            missing_files += 1

    print(f"\nProcess complete!")
    print(f"Successfully copied: {files_copied} files.")
    if missing_files > 0:
        print(f"Files not found: {missing_files}")

if __name__ == "__main__":
    main()
