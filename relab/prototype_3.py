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
NUM_BANDS = 12
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
    spectra = []
    names = []
    print("Loading pure mineral spectra...")
    for name, filename in MINERAL_FILES.items():
        spectra.append(load_and_interpolate(filename, WAVELENGTHS))
        names.append(name)
    return np.column_stack(spectra), names

# ==========================================
# 2. SENSOR PHYSICS MODELS
# ==========================================
def model_dynamic_fwhm(wls):
    slope = (400 - 10) / (40000 - 300)
    intercept = 10 - (slope * 300)
    return slope * wls + intercept

def model_sensor_noise(wls):
    # REALISTIC NOISE FIX: Variance scales from 1e-6 (VNIR) to 5e-5 (FIR)
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
    first_step_gain = None

    for step in range(num_bands):
        gains = np.zeros(num_wls)
        
        for i in range(num_wls):
            wl_candidate = wls[i]
            if any(abs(wl_candidate - wls[idx]) < MIN_SPACING_NM for idx in selected_indices):
                continue
                
            m_i = M[i, :]
            sigma2 = noise_var[i]
            
            gains[i] = np.dot(m_i.T, np.dot(F_inv, m_i)) / sigma2
            
        if step == 0:
            first_step_gain = gains.copy()
            
        best_idx = np.argmax(gains)
        selected_indices.append(best_idx)
        
        m_best = M[best_idx, :]
        sigma2_best = noise_var[best_idx]
        
        numerator = np.outer(np.dot(F_inv, m_best), np.dot(m_best.T, F_inv))
        denominator = sigma2_best + np.dot(m_best.T, np.dot(F_inv, m_best))
        F_inv = F_inv - (numerator / denominator)
        
    return selected_indices, first_step_gain

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_results(M_blurred, names, wls, fwhm_array, noise_var, selected_indices, gain_curve):
    print("Generating mission design visualization...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 1, figsize=(10, 6), 
                                        gridspec_kw={'height_ratios': [3, 1.5, 1]}, sharex=True)
    
    colors = ['blue', 'orange', 'gray', 'purple', 'brown', 'green']
    for j, name in enumerate(names):
        linewidth = 2 if name in ["Ilmenite", "Pyrite"] else 1
        alpha = 0.9 if name in ["Ilmenite", "Pyrite"] else 0.4
        ax1.plot(wls, M_blurred[:, j], label=name, color=colors[j], linewidth=linewidth, alpha=alpha)

    for idx in selected_indices:
        center_wl = wls[idx]
        band_width = fwhm_array[idx]
        ax1.axvspan(center_wl - band_width/2, center_wl + band_width/2, 
                    color='red', alpha=0.3, label='Optimal Sensor Band' if idx == selected_indices[0] else "")
        ax1.text(center_wl, ax1.get_ylim()[1]*0.95, f"{int(center_wl)}nm\n(Δ{int(band_width)}nm)", 
                 color='darkred', rotation=90, va='top', ha='center', fontweight='bold', fontsize=9)

    #ax1.set_title(f"1. Endmember Spectra & {len(selected_indices)} Optimal Sensor Bandwidths", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Simulated Sensor Signal", fontsize=12)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")
    ax1.grid(True, linestyle=':', alpha=0.6)

    '''
    ax2.plot(wls, gain_curve, color='navy', linewidth=1.5, label='Information Gain')
    for idx in selected_indices:
        ax2.axvline(x=wls[idx], color='red', alpha=0.5, linestyle='--', linewidth=2)
    ax2.set_title("2. Fisher Information Gain Landscape (1st Iteration)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Marginal Determinant Gain", fontsize=10)
    ax2.fill_between(wls, gain_curve, color='navy', alpha=0.1)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax3_twin = ax3.twinx()
    ax3.plot(wls, fwhm_array, color='darkgreen', linewidth=2, label='Bandwidth FWHM (nm)')
    ax3_twin.plot(wls, noise_var, color='darkred', linewidth=2, linestyle='-.', label='Noise Variance (σ²)')
    
    ax3.set_title("3. Modeled Sensor Constraints", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Wavelength (nm)", fontsize=12)
    ax3.set_ylabel("FWHM (nm)", color='darkgreen', fontsize=10)
    ax3_twin.set_ylabel("Noise Variance", color='darkred', fontsize=10)
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.grid(True, linestyle=':', alpha=0.6)
    '''

    plt.tight_layout()
    plt.savefig("provable_optimal_bands_ABSTRACT.png", dpi=150)
    plt.show()

def main():
    M_raw, mineral_names = build_endmember_matrix()
    fwhm_array = model_dynamic_fwhm(WAVELENGTHS)
    noise_var = model_sensor_noise(WAVELENGTHS)
    M_blurred = apply_spectral_response_function(M_raw, WAVELENGTHS, fwhm_array)
    selected_indices, gain_curve = submodular_greedy_selection(
        M_blurred, noise_var, WAVELENGTHS, NUM_BANDS
    )
    print("\nOptimization Complete.")
    print(f"Selected Band Centers (nm): {np.round(WAVELENGTHS[selected_indices], 1)}")
    plot_results(M_blurred, mineral_names, WAVELENGTHS, fwhm_array, noise_var, selected_indices, gain_curve)

if __name__ == "__main__":
    main()
