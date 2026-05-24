import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = Path("filtered_minerals")
PLOT_DIR = Path("mineral_plots")

# Create output directory for plots
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# The full list of target minerals for the report
TARGET_MINERALS = [
    "Ilmenite", "Pyrite", "Troilite", "Marcasite", "Chalcopyrite", 
    "Pyrrhotite", "Pigeonite", "Orthopyroxene", "Anorthite", "Fayalite", 
    "Hematite", "Magnetite", "Apatite", "Jarosite", "Alunite", 
    "Gypsum", "Anhydrite", "Microcline", "Prehnite", "Akaganeite", "Quartz"
]

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Directory '{INPUT_DIR}' not found.")
        return

    # Dictionary to map minerals to their respective files
    mineral_files = {mineral: [] for mineral in TARGET_MINERALS}
    
    # Scan the input directory and group files by mineral
    for filepath in INPUT_DIR.glob("*.*"):
        filename = filepath.name
        # Extract mineral name from our previously created naming convention
        mineral_name = filename.split('_')[0]
        
        # Match against target list (case-insensitive)
        for target in TARGET_MINERALS:
            if mineral_name.lower() == target.lower():
                mineral_files[target].append(filepath)
                break

    found_minerals = []
    missing_minerals = []

    print("Generating plots...\n")
    
    for mineral, files in mineral_files.items():
        if not files:
            missing_minerals.append(mineral)
            continue
            
        found_minerals.append(mineral)
        
        # Initialize a plot for this mineral
        plt.figure(figsize=(10, 6))
        
        for file in files:
            spectrum_id = file.stem.split('_')[-1] # Extract the spectrum ID for the legend
            
            try:
                # Read the file using pandas
                df = pd.read_csv(file, sep=r'\s+', skiprows=1, on_bad_lines='skip')
                
                if df.shape[1] >= 2:
                    # Extract columns and immediately convert them to 1D numpy arrays
                    wavelength_raw = df.iloc[:, 0].to_numpy()
                    reflectance_raw = df.iloc[:, 1].to_numpy()
                    
                    # Convert to numeric, forcing text/errors to NaN
                    wavelength = pd.to_numeric(wavelength_raw, errors='coerce')
                    reflectance = pd.to_numeric(reflectance_raw, errors='coerce')
                    
                    # Create a boolean mask to filter out NaN values using numpy
                    valid_mask = ~np.isnan(wavelength) & ~np.isnan(reflectance)
                    
                    # Plot using the cleaned numpy arrays
                    plt.plot(wavelength[valid_mask], reflectance[valid_mask], label=spectrum_id, linewidth=1.2)
            except Exception as e:
                print(f"  Warning: Could not parse {file.name} - {e}")

        # Plot formatting
        plt.title(f"Reflectance Spectra: {mineral}", fontsize=14, fontweight='bold')
        plt.xlabel("Wavelength (μm)", fontsize=12)
        plt.ylabel("Reflectance", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Place legend outside the plot area so it doesn't cover data
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()
        
        # Save and close the figure
        plot_path = PLOT_DIR / f"{mineral}_spectra.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Created plot: {plot_path.name} (from {len(files)} files)")

    # --- GENERATE THE REPORT ---
    print("\n" + "="*50)
    print("MINERAL SEARCH & PLOTTING REPORT")
    print("="*50)
    print(f"Total Target Minerals: {len(TARGET_MINERALS)}")
    print(f"Minerals Found:        {len(found_minerals)}")
    print(f"Minerals Missing:      {len(missing_minerals)}\n")
    
    if missing_minerals:
        print("Missing Minerals (No full-range files found in RELAB):")
        for m in missing_minerals:
            print(f"  - {m}")
    print("="*50)

if __name__ == "__main__":
    main()
