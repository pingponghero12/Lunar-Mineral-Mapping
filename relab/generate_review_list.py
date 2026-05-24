import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
CATALOG_DIR = Path("catalogues")
OUTPUT_CSV = "spectra_review_list.csv"

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

def main():
    print("Loading catalogues...")
    samples = pd.read_csv(CATALOG_DIR / "Sample_Catalogue.txt", sep='\t', encoding='latin1', on_bad_lines='skip', low_memory=False)
    spectra = pd.read_csv(CATALOG_DIR / "Spectra_Catalogue.txt", sep='\t', encoding='latin1', on_bad_lines='skip', low_memory=False)
    
    samples.columns = samples.columns.str.strip()
    spectra.columns = spectra.columns.str.strip()
    
    merged_df = pd.merge(spectra, samples, on="SampleID", how="inner")
    merged_df['Start'] = pd.to_numeric(merged_df['Start'], errors='coerce')
    merged_df['Stop'] = pd.to_numeric(merged_df['Stop'], errors='coerce')
    
    # Filter for wavelength range
    range_filtered = merged_df[(merged_df['Start'] <= 350) & (merged_df['Stop'] >= 24900)].copy()
    
    # List to store our curated rows
    curated_data = []

    for index, row in range_filtered.iterrows():
        sample_name_desc = str(row.get('SampleName', '')).lower()
        
        for mineral_name, search_terms in TARGET_MINERALS.items():
            if any(term in sample_name_desc for term in search_terms):
                # Extract only the most useful columns for manual review
                curated_data.append({
                    "Target_Mineral": mineral_name,
                    "SpectrumID": str(row['SpectrumID']).strip(),
                    "SampleID": str(row['SampleID']).strip(),
                    "PI_Code": str(row['PI']).strip(),
                    "SampleName": str(row['SampleName']).strip(),
                    "MinSize_um": row.get('MinSize', ''),
                    "MaxSize_um": row.get('MaxSize', ''),
                    "Texture": row.get('Texture', ''),
                    "Origin": row.get('Origin', ''),
                })
                break # Move to next row once matched

    # Create a DataFrame and save to CSV
    output_df = pd.DataFrame(curated_data)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Extracted {len(output_df)} entries to '{OUTPUT_CSV}'.")
    print("Please open this CSV, delete the rows/spectra you do NOT want, and save it.")

if __name__ == "__main__":
    main()
