import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from pathlib import Path

# --- INPUTS ---
INPUT_DIR = Path("filtered_minerals")
# Replace this array with the exact output from Script 1
#CHOSEN_BANDS = np.array([572., 2725.8, 5931.7, 24742.5, 9030.5, 15305.4, 7432.5, 4361.5, 10997.7, 16806.1, 21060.2, 18306.8])
#CHOSEN_BANDS = np.array([ 601.7,   300.,  2723.8, 1771.,   959.1, 7756.,  2360.5, 3061.3,  522.3,  887.6, 315.9, 1697.5])
#CHOSEN_BANDS = np.array([ 601.7,  300.,  2723.8, 1771.,   959.1])
CHOSEN_BANDS = np.array([595.8,  2765.5, 24744.5,  3061.3,  1538.7,   689.1,  3045.4, 24728.6])
#

MINERAL_FILES = {
    "Ilmenite": "Ilmenite_BKR1LR222.txt",
    "Pyrite": "Pyrite_BKR1ZL014.txt",
    "Anorthite": "Anorthite_BKR1DH001.txt",
    "Pigeonite": "Pigeonite_BKR1DD055.txt",
    "Orthopyroxene": "Orthopyroxene_BKR1PE011.txt",
    "Fayalite": "Fayalite_BKR1OL018A.txt"
}

# 1. Physics Models (Imported from selection script to ensure mathematical parity)
def model_dynamic_fwhm(wls):
    slope = (400 - 10) / (40000 - 300)
    intercept = 10 - (slope * 300)
    return slope * wls + intercept

def load_full_spectrum(filename, full_wls):
    filepath = INPUT_DIR / filename
    try:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, on_bad_lines='skip')
        wl = pd.to_numeric(df.iloc[:, 0].to_numpy(), errors='coerce')
        refl = pd.to_numeric(df.iloc[:, 1].to_numpy(), errors='coerce')
        mask = ~np.isnan(wl) & ~np.isnan(refl)
        wl, refl = wl[mask], refl[mask]
        
        if wl.max() < 100: wl = wl * 1000 
        sort_idx = np.argsort(wl)
        return np.interp(full_wls, wl[sort_idx], refl[sort_idx])
    except:
        return np.zeros_like(full_wls)

def apply_srf_to_chosen_bands(M_raw, full_wls, chosen_wls):
    """Applies the FWHM blur specifically targeting our chosen bands."""
    fwhm_array = model_dynamic_fwhm(chosen_wls)
    M_s = np.zeros((len(chosen_wls), M_raw.shape[1]))
    
    for i, wl in enumerate(chosen_wls):
        fwhm = fwhm_array[i]
        sigma = fwhm / 2.355
        diff = full_wls - wl
        mask = np.abs(diff) < 4 * sigma
        gauss = np.exp(-(diff[mask]**2) / (2 * sigma**2))
        gauss_normalized = gauss / np.sum(gauss)
        M_s[i, :] = np.dot(gauss_normalized, M_raw[mask, :])
    return M_s

def main():
    print(f"Initializing Sensor at Bands: {CHOSEN_BANDS} nm")
    
    # 2. Build High-Res Endmembers
    full_wls = np.linspace(300, 40000, 20000)
    pure_ilm = load_full_spectrum(MINERAL_FILES["Ilmenite"], full_wls)
    pure_pyr = load_full_spectrum(MINERAL_FILES["Pyrite"], full_wls)
    ano = load_full_spectrum(MINERAL_FILES["Anorthite"], full_wls)
    pig = load_full_spectrum(MINERAL_FILES["Pigeonite"], full_wls)
    opx = load_full_spectrum(MINERAL_FILES["Orthopyroxene"], full_wls)
    fay = load_full_spectrum(MINERAL_FILES["Fayalite"], full_wls)

    basalt = (0.30 * ano) + (0.25 * pig) + (0.25 * opx) + (0.20 * fay)
    M_raw = np.column_stack([basalt, pure_ilm, pure_pyr])
    names = ["Lunar Basalt", "Ilmenite", "Pyrite"]
    
    # 3. Apply the Sensor Optics (The "Bucket" Fix)
    M_s = apply_srf_to_chosen_bands(M_raw, full_wls, CHOSEN_BANDS)
    
    # 4. Simulate Sensor Noise (Realistic SNR Fix)
    slope = (0.00005 - 0.000001) / (40000 - 300)
    intercept = 0.000001 - (slope * 300)
    noise_variances = slope * CHOSEN_BANDS + intercept
    Sigma_inv = np.diag(1.0 / noise_variances)

    # 5. Calculate Theoretical Confidence
    FIM = np.dot(M_s.T, np.dot(Sigma_inv, M_s))
    Covariance = np.linalg.inv(FIM)
    standard_errors = np.sqrt(np.diag(Covariance))

    # 6. Generate Ground Truth
    true_abundances = np.array([0.75, 0.18, 0.07])
    clean_signal = np.dot(M_s, true_abundances)
    noise = np.random.normal(0, np.sqrt(noise_variances))
    measured_signal = clean_signal + noise

    # 7. Unmix (Removed forced 100% normalization)
    estimated_abundances, residual = nnls(M_s, measured_signal)

    # 8. Print Results
    print("\n--- UNMIXING RESULTS ---")
    for i, name in enumerate(names):
        true_pct = true_abundances[i] * 100
        est_pct = estimated_abundances[i] * 100
        err_pct = standard_errors[i] * 100
        print(f"{name:15}: True = {true_pct:5.1f}% | Sensor Estimate = {est_pct:5.1f}% ± {err_pct:4.1f}%")

    # 9. Visualization
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, true_abundances * 100, width, label='True Composition', color='darkgray')
    ax.bar(x + width/2, estimated_abundances * 100, width, 
           yerr=standard_errors * 100, capsize=8, 
           label='Sensor Unmixed Estimate', color=['gray', 'blue', 'orange'], alpha=0.8)

    ax.set_ylabel('Abundance (%)', fontsize=12)
    ax.set_title(f'Linear Spectral Unmixing Confidence\n(Using {len(CHOSEN_BANDS)} Optimal FIR Bands)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("unmixing_confidence.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
