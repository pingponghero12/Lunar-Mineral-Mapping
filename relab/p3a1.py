import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = Path("filtered_minerals")
WAVELENGTHS = np.linspace(300, 40000, 20000) # 20,000 points from 0.3 to 40 µm
NUM_BANDS = 8
MIN_SPACING_NM = 15

MINERAL_FILES = {
    "Ilmenite": "Ilmenite_BKR1LR222.txt",
    "Pyrite": "Pyrite_BKR1ZL014.txt",
    "Anorthite": "Anorthite_BKR1DH001.txt",
    "Pigeonite": "Pigeonite_BKR1DD055.txt",
    "Orthopyroxene": "Orthopyroxene_BKR1PE011.txt",
    "Fayalite": "Fayalite_BKR1OL018A.txt"
}

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_interpolate(filename, target_wls):
    filepath = INPUT_DIR / filename
    try:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, on_bad_lines='skip')
        if df.shape[1] >= 2:
            wl_raw = pd.to_numeric(df.iloc[:, 0].to_numpy(), errors='coerce')
            refl_raw = pd.to_numeric(df.iloc[:, 1].to_numpy(), errors='coerce')
            
            mask = ~np.isnan(wl_raw) & ~np.isnan(refl_raw)
            wls, refl = wl_raw[mask], refl_raw[mask]
            
            if wls.max() < 100:
                wls = wls * 1000
                
            sort_idx = np.argsort(wls)
            return np.interp(target_wls, wls[sort_idx], refl[sort_idx])
    except Exception as e:
        print(f"Warning: Could not parse {filename} - {e}")
    return np.zeros_like(target_wls)

def build_endmember_matrix():
    print("Loading pure mineral spectra and creating Basalt mix...")
    # Load individual spectra
    ano = load_and_interpolate(MINERAL_FILES["Anorthite"], WAVELENGTHS)
    pig = load_and_interpolate(MINERAL_FILES["Pigeonite"], WAVELENGTHS)
    opx = load_and_interpolate(MINERAL_FILES["Orthopyroxene"], WAVELENGTHS)
    fay = load_and_interpolate(MINERAL_FILES["Fayalite"], WAVELENGTHS)
    
    pure_ilm = load_and_interpolate(MINERAL_FILES["Ilmenite"], WAVELENGTHS)
    pure_pyr = load_and_interpolate(MINERAL_FILES["Pyrite"], WAVELENGTHS)

    # Group background into one "Basalt" endmember
    basalt = (0.30 * ano) + (0.25 * pig) + (0.25 * opx) + (0.20 * fay)

    spectra = [basalt, pure_ilm, pure_pyr]
    names = ["Lunar Basalt", "Ilmenite", "Pyrite"]
    
    return np.column_stack(spectra), names

# ==========================================
# 2. SENSOR PHYSICS MODELS
# ==========================================
def model_dynamic_fwhm(wls):
    slope = (400 - 10) / (40000 - 300)
    intercept = 10 - (slope * 300)
    return slope * wls + intercept

def model_sensor_noise(wls):
    # REALISTIC NOISE: Variance scales from 1e-6 (VNIR) to 5e-5 (FIR)
    slope = (0.00005 - 0.000001) / (40000 - 300)
    intercept = 0.000001 - (slope * 300)
    return slope * wls + intercept

def apply_spectral_response_function(M_raw, wls, fwhm_array):
    print("Applying dynamic Spectral Response Function (blurring)...")
    M_blurred = np.zeros_like(M_raw)
    
    for i, wl in enumerate(wls):
        fwhm = fwhm_array[i]
        sigma = fwhm / 2.355
        
        diff = wls - wl
        mask = np.abs(diff) < 4 * sigma
        
        gauss = np.exp(-(diff[mask]**2) / (2 * sigma**2))
        gauss_normalized = gauss / np.sum(gauss)
        
        M_blurred[i, :] = np.dot(gauss_normalized, M_raw[mask, :])
        
    return M_blurred

# ==========================================
# 3. INFORMATION THEORY OPTIMIZATION
# ==========================================
def submodular_greedy_selection(M, noise_var, wls, num_bands):
    print("Running D-Optimal Greedy Selection (Information Theory)...")
    num_wls, num_minerals = M.shape
    
    F_inv = np.eye(num_minerals) * 1e6 
    selected_indices = []

    for step in range(num_bands):
        gains = np.zeros(num_wls)
        
        for i in range(num_wls):
            wl_candidate = wls[i]
            if any(abs(wl_candidate - wls[idx]) < MIN_SPACING_NM for idx in selected_indices):
                continue
                
            m_i = M[i, :]
            sigma2 = noise_var[i]
            
            gains[i] = np.dot(m_i.T, np.dot(F_inv, m_i)) / sigma2
            
        best_idx = np.argmax(gains)
        selected_indices.append(best_idx)
        
        m_best = M[best_idx, :]
        sigma2_best = noise_var[best_idx]
        
        numerator = np.outer(np.dot(F_inv, m_best), np.dot(m_best.T, F_inv))
        denominator = sigma2_best + np.dot(m_best.T, np.dot(F_inv, m_best))
        F_inv = F_inv - (numerator / denominator)
        
    return selected_indices

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_results(M_blurred, names, wls, fwhm_array, selected_indices):
    print("Generating mission design visualization...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colors matching the unmixing script (Basalt, Ilmenite, Pyrite)
    colors = ['gray', 'blue', 'orange'] 
    
    for j, name in enumerate(names):
        linewidth = 2 if name in ["Ilmenite", "Pyrite"] else 1.5
        alpha = 0.9 if name in ["Ilmenite", "Pyrite"] else 0.6
        ax.plot(wls, M_blurred[:, j], label=name, color=colors[j], linewidth=linewidth, alpha=alpha)

    for idx in selected_indices:
        center_wl = wls[idx]
        band_width = fwhm_array[idx]
        ax.axvspan(center_wl - band_width/2, center_wl + band_width/2, 
                    color='red', alpha=0.3, label='Optimal Sensor Band' if idx == selected_indices[0] else "")
        ax.text(center_wl, ax.get_ylim()[1]*0.95, f"{int(center_wl)}nm\n(Δ{int(band_width)}nm)", 
                 color='darkred', rotation=90, va='top', ha='center', fontweight='bold', fontsize=9)

    ax.set_title(f"Endmember Spectra & {len(selected_indices)} Optimal Sensor Bandwidths", fontsize=14, fontweight='bold')
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Simulated Sensor Signal", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("provable_optimal_bands_FIR.png", dpi=150)
    plt.show()

def main():
    M_raw, mineral_names = build_endmember_matrix()
    fwhm_array = model_dynamic_fwhm(WAVELENGTHS)
    noise_var = model_sensor_noise(WAVELENGTHS)
    M_blurred = apply_spectral_response_function(M_raw, WAVELENGTHS, fwhm_array)
    selected_indices = submodular_greedy_selection(
        M_blurred, noise_var, WAVELENGTHS, NUM_BANDS
    )
    print("\nOptimization Complete.")
    print(f"Selected Band Centers (nm): {np.round(WAVELENGTHS[selected_indices], 1)}")
    plot_results(M_blurred, mineral_names, WAVELENGTHS, fwhm_array, selected_indices)

if __name__ == "__main__":
    main()
