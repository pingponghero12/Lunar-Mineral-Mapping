import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_DIR = Path("filtered_minerals")

# Upgraded to the full FIR scope: 300 nm to 40,000 nm (40 microns) with 20,000 points
common_wavelengths = np.linspace(300, 40000, 20000) 

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

def get_local_maxima(array):
    """Finds indices of local maxima in a 1D numpy array."""
    # True where the current value is greater than both neighbors
    return (np.diff(np.sign(np.diff(array))) < 0).nonzero()[0] + 1

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

    # Prevent division by zero in contrast calculation
    safe_bg = np.where(bg_spectrum <= 0, 1e-9, bg_spectrum)

    # --- UPGRADED ALGORITHM LOGIC ---
    mix_ilmenite = 0.90 * bg_spectrum + 0.10 * pure_ilmenite
    mix_pyrite = 0.90 * bg_spectrum + 0.10 * pure_pyrite
    
    # 1. Calculate Relative Contrast
    contrast_ilm = np.abs(mix_ilmenite - safe_bg) / safe_bg
    contrast_pyr = np.abs(mix_pyrite - safe_bg) / safe_bg
    total_contrast = contrast_ilm + contrast_pyr

    # 2. Find Local Maxima (Peaks of the contrast features)
    local_max_indices = get_local_maxima(total_contrast)
    
    # 3. Sort only the peaks by their contrast magnitude
    peak_values = total_contrast[local_max_indices]
    sorted_peak_indices = local_max_indices[np.argsort(peak_values)[::-1]]

    num_bands = 5
    # Enforce a wider spacing because the range is massive (40,000 nm)
    min_spacing_nm = 1500 
    chosen_indices = []

    for idx in sorted_peak_indices:
        if len(chosen_indices) >= num_bands: break
        wl_candidate = common_wavelengths[idx]
        
        # Check minimum spacing constraint
        if all(abs(wl_candidate - common_wavelengths[c_idx]) >= min_spacing_nm for c_idx in chosen_indices):
            chosen_indices.append(idx)

    chosen_wavelengths = common_wavelengths[chosen_indices]

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Plot 1: The Spectra
    ax1.plot(common_wavelengths, bg_spectrum, label="Simulated Basalt (Background)", color="black", linewidth=1.5, linestyle="--")
    ax1.plot(common_wavelengths, pure_ilmenite, label="Pure Ilmenite", color="blue", alpha=0.8)
    ax1.plot(common_wavelengths, pure_pyrite, label="Pure Pyrite", color="orange", alpha=0.8)
    
    ax1.set_title("Band Selection via Relative Contrast & Peak Detection (FIR Range)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Reflectance / Emissivity", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: The Contrast Signal (to show the algorithm's "brain")
    ax2.plot(common_wavelengths, total_contrast, label="Combined Relative Contrast Signal", color="purple", linewidth=1.2)
    ax2.set_ylabel("Contrast Magnitude", fontsize=12)
    ax2.set_xlabel("Wavelength (nm)", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Highlight Chosen Bands on both plots
    for i, wl in enumerate(chosen_wavelengths):
        label = "Optimal Band Center" if i == 0 else ""
        ax1.axvline(x=wl, color='red', alpha=0.4, linewidth=2.5, label=label)
        ax2.axvline(x=wl, color='red', alpha=0.4, linewidth=2.5)
        
        # Add text labels at the top of the upper plot
        ax1.text(wl, ax1.get_ylim()[1]*0.95, f"{int(wl)} nm", color='red', rotation=90, va='top', ha='right', fontweight='bold')

    # Fix legend duplication for vlines
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()

    print(f"\nOptimization complete across 20,000 points.")
    print(f"Selected Bands Centers: {np.round(chosen_wavelengths, 1)} nm")
    
    plt.savefig("advanced_band_selection_FIR.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
