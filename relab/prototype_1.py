import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_DIR = Path("filtered_minerals")
common_wavelengths = np.linspace(400, 40000, 20000) 

def load_and_interpolate(filename):
    """Loads a 2-column txt file using pandas and interpolates to a common grid."""
    filepath = INPUT_DIR / filename
    try:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, on_bad_lines='skip')
        
        if df.shape[1] >= 2:
            wavelength_raw = df.iloc[:, 0].to_numpy()
            reflectance_raw = df.iloc[:, 1].to_numpy()
            
            wavelength = pd.to_numeric(wavelength_raw, errors='coerce')
            reflectance = pd.to_numeric(reflectance_raw, errors='coerce')
            
            valid_mask = ~np.isnan(wavelength) & ~np.isnan(reflectance)
            wls = wavelength[valid_mask]
            refl = reflectance[valid_mask]
            
            # Auto-detect if data is in microns and convert to nm
            if wls.max() < 100:
                wls = wls * 1000
                
            sort_idx = np.argsort(wls)
            wls, refl = wls[sort_idx], refl[sort_idx]
            
            return np.interp(common_wavelengths, wls, refl)
        return np.zeros_like(common_wavelengths)
    except Exception as e:
        print(f"  Warning: Could not parse {filename} - {e}")
        return np.zeros_like(common_wavelengths)

def main():
    print("Loading pure mineral spectra...")
    
    # Pure Targets
    pure_ilmenite = load_and_interpolate("Ilmenite_BKR1LR222.txt")
    pure_pyrite = load_and_interpolate("Pyrite_BKR1ZL014.txt")

    # Basalt Background Components
    anorthite = load_and_interpolate("Anorthite_BKR1DH001.txt")
    pigeonite = load_and_interpolate("Pigeonite_BKR1DD055.txt")
    orthopyroxene = load_and_interpolate("Orthopyroxene_BKR1PE011.txt")
    fayalite = load_and_interpolate("Fayalite_BKR1OL018A.txt")

    # Create Background
    bg_spectrum = (0.30 * anorthite) + (0.25 * pigeonite) + (0.25 * orthopyroxene) + (0.20 * fayalite)

    # --- ALGORITHM LOGIC (Still based on 10% mix for realistic sensitivity) ---
    mix_ilmenite = 0.90 * bg_spectrum + 0.10 * pure_ilmenite
    mix_pyrite = 0.90 * bg_spectrum + 0.10 * pure_pyrite
    
    total_separability = np.abs(mix_ilmenite - bg_spectrum) + np.abs(mix_pyrite - bg_spectrum)

    num_bands = 5
    min_spacing_nm = 150 # Slightly wider spacing for better visualization
    chosen_indices = []
    sorted_indices = np.argsort(total_separability)[::-1]

    for idx in sorted_indices:
        if len(chosen_indices) >= num_bands: break
        wl_candidate = common_wavelengths[idx]
        if all(abs(wl_candidate - common_wavelengths[c_idx]) >= min_spacing_nm for c_idx in chosen_indices):
            chosen_indices.append(idx)

    chosen_wavelengths = common_wavelengths[chosen_indices]

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 7))

    # Plot Pure Endmembers
    plt.plot(common_wavelengths, bg_spectrum, label="Simulated Basalt (Background)", color="black", linewidth=2, linestyle="--")
    plt.plot(common_wavelengths, pure_ilmenite, label="Pure Ilmenite", color="blue", alpha=0.8)
    plt.plot(common_wavelengths, pure_pyrite, label="Pure Pyrite", color="orange", alpha=0.8)

    # Highlight Chosen Bands
    for i, wl in enumerate(chosen_wavelengths):
        plt.axvline(x=wl, color='red', alpha=0.3, linewidth=3, label="Selected Optimal Band" if i == 0 else "")
        plt.text(wl, plt.ylim()[1]*0.9, f"{int(wl)}nm", color='red', rotation=90, va='top', fontweight='bold')

    plt.title("Band Selection Analysis: Pure Endmembers vs. Basalt Background", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    print(f"\nOptimization complete.")
    print(f"Selected Bands: {np.round(chosen_wavelengths, 1)} nm")
    
    plt.savefig("pure_mineral_band_selection.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
