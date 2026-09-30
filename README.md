# Lunar spectral baseline design

This repository contains the reproducible study for **Information-Theoretic
Spectral Baseline Design for In-Situ Lunar Resource Mapping**. It selects small
sets of candidate spectral bands from measured RELAB spectra and tests how the
result changes with band count, assumed noise, filter spacing, and simple model
mismatch.

The project is intentionally limited. It is a laboratory-domain instrument
baseline study, not a flight-instrument model or an orbital mineral map.

## Method in brief

1. Load the selected RELAB spectra and determine their common measured support.
2. Convolve each spectrum with wavelength-dependent Gaussian band responses.
3. Whiten component contrasts using one of three stated noise assumptions.
4. Select bands by greedy D-optimal design under a configurable spacing rule;
5. compare the reference solution with a small fixed-seed genetic search; and
6. test abundance recovery using noisy synthetic mixtures, non-negative least
   squares, and projection onto the abundance simplex.

No spectrum is extrapolated. A case stops where any required constituent stops.
The four study cases are defined in [`configs/study.yaml`](configs/study.yaml):

- a five-band mare-resource example;
- a six-component set with genuine measured support to 40 micrometres; and
- seven functional groups covering all 17 priority minerals; and
- a 20-band diagnostic retaining all 17 minerals as separate endmembers.

## Reproduce the study

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_study.py
pytest
```

The run is deterministic. It writes machine-readable tables to
`outputs/results`, spectrum checksums and wavelength support to
`outputs/provenance`, paper figures to `outputs/figures`, and LaTeX result
macros to `paper/results.tex`.

Build the APISAT-formatted manuscript with:

```bash
bash scripts/build_paper.sh
```

The compressed submission file is written to `paper/submission.pdf`.

## Main outputs

- `outputs/results/design_sweep.csv`: band-count, noise, and spacing sweep;
- `outputs/results/selected_bands.csv`: every selected band and adopted FWHM;
- `outputs/results/unmixing_stress.csv`: nominal and stressed mixture errors;
- `outputs/results/genetic_audit.json`: limited greedy/genetic comparison;
- `outputs/provenance/spectra.json`: exact inputs, support, and SHA-256 hashes;
- `outputs/figures`: all manuscript plots; and
- `paper/main.tex`: complete manuscript source.

## Interpretation and limitations

The MIR and FIR noise levels are sensitivity assumptions because an at-sensor
radiance model and a particular detector have not been selected. Laboratory
reflectance does not include lunar temperature, illumination, particulate
mixing, grain-size effects, space weathering, or spatial detectability. The
eight-band all-priority case estimates seven functional groups, while the
mineral-specific diagnostic uses 20 bands to estimate all 17 abundances. The
small genetic search is a cross-check, not a proof of global optimality. These
boundaries are discussed explicitly in the manuscript.

The older scripts under `relab/` are retained for provenance. The packaged
workflow under `lunar_band_design/` is the implementation used for the reported
results.
