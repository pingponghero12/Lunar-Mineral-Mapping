import os
import shutil
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
# Set your paths relative to where this script is running
CATALOG_DIR = Path("catalogues")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("filtered_minerals")

# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define our targets and their search terms (lowercase for easy matching)
TARGET_MINERALS = {
    "Ilmenite": ["ilmenite", "titanium iron oxide", "opaque mineral"],
    "Pyrite": ["pyrite", "iron disulfide", "cubic iron sulfide"],
    "Troilite": ["troilite", "iron monosulfide", "stoichiometric iron sulfide"],
    "Marcasite": ["marcasite", "white iron pyrite", "orthorhombic iron sulfide"],
    "Chalcopyrite": ["chalcopyrite", "copper iron sulfide"],
    "Pyrrhotite": ["pyrrhotite", "magnetic pyrite"],
    "Pigeonite": ["pigeonite", "clinopyroxene", "cpx", "low-calcium clinopyroxene"],
    "Orthopyroxene": ["orthopyroxene", "enstatite", "ferrosilite", "opx"],
    "Anorthite": ["anorthite", "labradorite", "bytownite", "plagioclase"],
    "Fayalite": ["fayalite", "forsterite", "olivine"],
    "Hematite": ["hematite", "ferric oxide", "specularite"],
    "Magnetite": ["magnetite", "magnetic iron ore"],
    "Apatite": ["apatite", "calcium phosphate", "fluorapatite", "hydroxylapatite"],
    "Jarosite": ["jarosite", "hydrous iron sulfate"],
    "Alunite": ["alunite", "alum stone"],
    "Gypsum": ["gypsum", "hydrated calcium sulfate", "selenite"],
    "Anhydrite": ["anhydrite", "anhydrous calcium sulfate"],
    "Microcline": ["microcline", "alkali feldspar", "k-spar"],
    "Prehnite": ["prehnite", "hydrated calcium aluminum silicate"],
    "Akaganeite": ["akaganeite", "beta-iron oxyhydroxide"],
    "Quartz": ["quartz", "silica"]
}

def load_catalogues():
    print("Loading catalogues...")
    # Fix: Added encoding='latin1' to handle the μ symbols and other special characters
    samples = pd.read_csv(CATALOG_DIR / "Sample_Catalogue.txt", sep='\t', encoding='latin1', on_bad_lines='skip', low_memory=False)
    spectra = pd.read_csv(CATALOG_DIR / "Spectra_Catalogue.txt", sep='\t', encoding='latin1', on_bad_lines='skip', low_memory=False)
    
    # Clean up column names (strip trailing/leading spaces)
    samples.columns = samples.columns.str.strip()
    spectra.columns = spectra.columns.str.strip()
    
    return samples, spectra

def main():
    samples, spectra = load_catalogues()
    
    # Merge the two dataframes on SampleID
    merged_df = pd.merge(spectra, samples, on="SampleID", how="inner")
    
    # Convert Start and Stop to numeric to allow for filtering
    merged_df['Start'] = pd.to_numeric(merged_df['Start'], errors='coerce')
    merged_df['Stop'] = pd.to_numeric(merged_df['Stop'], errors='coerce')
    
    # 1. Filter for Full Range (<= 350 nm start to allow wiggle room, and >= 25000 nm stop)
    range_filtered = merged_df[(merged_df['Start'] <= 450) & (merged_df['Stop'] >= 24900)]
    print(f"Found {len(range_filtered)} spectra with the required wavelength range.")

    files_copied = 0

    # 2. Iterate through our filtered list and match minerals
    for index, row in range_filtered.iterrows():
        sample_name_desc = str(row.get('SampleName', '')).lower()
        
        # Check if this sample matches any of our target minerals
        matched_mineral = None
        for mineral_name, search_terms in TARGET_MINERALS.items():
            if any(term in sample_name_desc for term in search_terms):
                matched_mineral = mineral_name
                break
        
        if matched_mineral:
            # We found a match! Now, let's construct the RELAB file path.
            # Rule: data / pi_code / first_two_letters_of_sampleID / spectrumID.txt
            pi_code = str(row['PI']).strip().lower()
            sample_id = str(row['SampleID']).strip().lower()
            first_two = sample_id[:2]
            spectrum_id = str(row['SpectrumID']).strip().lower()
            
            source_file = DATA_DIR / pi_code / first_two / f"{spectrum_id}.txt"
            
            if source_file.exists():
                # Create a human-readable name: e.g., Ilmenite_BIR1AA001.txt
                new_filename = f"{matched_mineral}_{spectrum_id.upper()}.txt"
                dest_file = OUTPUT_DIR / new_filename
                
                # Copy the file
                shutil.copy2(source_file, dest_file)
                print(f"Copied: {source_file.name} -> {new_filename}")
                files_copied += 1
            else:
                # Fallback to .asc if .txt doesn't exist
                source_file_asc = DATA_DIR / pi_code / first_two / f"{spectrum_id}.asc"
                if source_file_asc.exists():
                    new_filename = f"{matched_mineral}_{spectrum_id.upper()}.asc"
                    dest_file = OUTPUT_DIR / new_filename
                    shutil.copy2(source_file_asc, dest_file)
                    print(f"Copied: {source_file_asc.name} -> {new_filename}")
                    files_copied += 1
                else:
                    print(f"Warning: File not found for {spectrum_id} at {source_file.parent}")

    print(f"\nProcess complete! Successfully extracted and renamed {files_copied} files into '{OUTPUT_DIR.name}/'.")

if __name__ == "__main__":
    main()
