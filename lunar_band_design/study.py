"""Run the complete, intentionally small band-design study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .plots import plot_case_designs, plot_sensitivity, plot_spacing, plot_unmixing
from .selection import contrast_vectors, genetic_select, greedy_select
from .sensor import fwhm_um, noise_sigma, valid_candidate_centers
from .spectra import build_components, convolve_bands
from .unmixing import monte_carlo_unmixing


def _linearised_metrics(vectors: np.ndarray, indices: tuple[int, ...]) -> dict[str, float | int]:
    selected = vectors[list(indices)]
    fisher = selected.T @ selected
    rank = int(np.linalg.matrix_rank(fisher))
    dimension = fisher.shape[0]
    if rank < dimension:
        return {"rank": rank, "dimension": dimension, "mean_crlb": float("inf"), "condition": float("inf")}
    covariance = np.linalg.inv(fisher)
    return {
        "rank": rank,
        "dimension": dimension,
        "mean_crlb": float(np.mean(np.sqrt(np.maximum(np.diag(covariance), 0.0)))),
        "condition": float(np.linalg.cond(fisher)),
    }


def _write_macros(path: Path, summary: dict) -> None:
    cases = summary["reference_cases"]
    lines = [
        rf"\newcommand{{\StudyCaseCount}}{{{len(cases)}}}",
        rf"\newcommand{{\StudySpectrumCount}}{{{summary['unique_spectrum_count']}}}",
    ]
    for key, prefix in [
        ("mare_resources", "Mare"),
        ("extended_longwave", "Longwave"),
        ("all_priority_groups", "AllGroups"),
    ]:
        case = cases[key]
        centers = ", ".join(f"{value:.2f}" for value in case["centers_um"])
        lines.extend(
            [
                rf"\newcommand{{\{prefix}Range}}{{{case['range_um'][0]:.2f}--{case['range_um'][1]:.2f}}}",
                rf"\newcommand{{\{prefix}BandCount}}{{{case['band_count']}}}",
                rf"\newcommand{{\{prefix}Bands}}{{{centers}}}",
                rf"\newcommand{{\{prefix}NominalRMSE}}{{{100*case['nominal_rmse']:.2f}}}",
                rf"\newcommand{{\{prefix}CombinedRMSE}}{{{100*case['combined_rmse']:.2f}}}",
                rf"\newcommand{{\{prefix}GreedyGap}}{{{100*case['ga_gap_fraction']:.2f}}}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def run(project_root: Path, config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    spectra_dir = project_root / "relab" / "filtered_minerals"
    results_dir = project_root / "outputs" / "results"
    figures_dir = project_root / "outputs" / "figures"
    provenance_dir = project_root / "outputs" / "provenance"
    for directory in [results_dir, figures_dir, provenance_dir, project_root / "paper"]:
        directory.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    band_rows: list[dict] = []
    monte_rows: list[dict] = []
    ga_rows: list[dict] = []
    provenance: dict[str, dict] = {}
    figure_payloads: list[dict] = []
    reference_cases: dict[str, dict] = {}

    for case_number, (case_key, case_config) in enumerate(config["cases"].items()):
        requested = (
            float(config["analysis_range_um"][0]),
            min(float(config["analysis_range_um"][1]), float(case_config["max_wavelength_um"])),
        )
        grid, component_spectra, component_names, loaded = build_components(
            case_config["components"],
            spectra_dir,
            float(config["spectral_grid_step_um"]),
            requested,
        )
        for spectrum in loaded:
            provenance[spectrum.path.name] = {
                "file": str(spectrum.path.relative_to(project_root)),
                "minimum_wavelength_um": spectrum.support[0],
                "maximum_wavelength_um": spectrum.support[1],
                "sample_count": int(spectrum.wavelength_um.size),
                "sha256": spectrum.sha256,
            }
        centers, widths = valid_candidate_centers(
            float(grid[0]), float(grid[-1]), float(config["candidate_step_um"])
        )
        candidate_response = convolve_bands(grid, component_spectra, centers, widths)

        selections: dict[tuple[str, int, float], object] = {}
        for noise_name, noise_profile in config["noise_profiles"].items():
            sigma = noise_sigma(candidate_response, centers, noise_profile)
            vectors = contrast_vectors(candidate_response, sigma)
            for band_count in config["band_counts"]:
                for spacing in config["spacing_factors"]:
                    result = greedy_select(
                        vectors, centers, widths, int(band_count), float(spacing)
                    )
                    selections[(noise_name, int(band_count), float(spacing))] = result
                    linear = _linearised_metrics(vectors, result.indices)
                    sweep_rows.append(
                        {
                            "case": case_key,
                            "noise_profile": noise_name,
                            "band_count": int(band_count),
                            "spacing_factor": float(spacing),
                            "objective": result.objective,
                            **linear,
                        }
                    )
                    for position, index in enumerate(result.indices, start=1):
                        band_rows.append(
                            {
                                "case": case_key,
                                "noise_profile": noise_name,
                                "band_count": int(band_count),
                                "spacing_factor": float(spacing),
                                "band_position": position,
                                "center_um": float(centers[index]),
                                "fwhm_um": float(widths[index]),
                            }
                        )

        reference_count = int(case_config.get("reference_band_count", config["reference_band_count"]))
        reference_spacing = float(config["reference_spacing_factor"])
        reference = selections[("reference", reference_count, reference_spacing)]
        selected_indices = np.array(reference.indices, dtype=int)
        selected_centers = centers[selected_indices]
        selected_widths = widths[selected_indices]
        model_response = candidate_response[selected_indices]

        reference_sigma = noise_sigma(
            model_response, selected_centers, config["noise_profiles"]["reference"]
        )
        reference_vectors = contrast_vectors(model_response, reference_sigma)
        ga = genetic_select(
            contrast_vectors(candidate_response, noise_sigma(candidate_response, centers, config["noise_profiles"]["reference"])),
            centers,
            widths,
            reference_count,
            reference_spacing,
            int(config["seed"]) + case_number,
        )
        gap_fraction = (reference.objective - ga.objective) / max(abs(reference.objective), 1e-12)
        ga_rows.append(
            {
                "case": case_key,
                "greedy_objective": reference.objective,
                "genetic_objective": ga.objective,
                "greedy_gap_fraction": gap_fraction,
                "greedy_centers_um": selected_centers.tolist(),
                "genetic_centers_um": centers[list(ga.indices)].tolist(),
            }
        )

        case_monte: dict[str, dict] = {}
        for stress_number, (stress_name, stress) in enumerate(config["stress_tests"].items()):
            rng = np.random.default_rng(int(config["seed"]) + 100 * case_number + stress_number)
            shift = rng.normal(0.0, float(stress["center_shift_fwhm"]), selected_centers.size) * selected_widths
            actual_centers = np.clip(
                selected_centers + shift,
                grid[0] + 1.70 * selected_widths,
                grid[-1] - 1.70 * selected_widths,
            )
            actual_response = convolve_bands(
                grid, component_spectra, actual_centers, selected_widths
            )
            sigma = noise_sigma(
                model_response,
                selected_centers,
                config["noise_profiles"][stress["noise_profile"]],
            )
            metrics = monte_carlo_unmixing(
                model_response,
                actual_response,
                sigma,
                np.asarray(case_config["dirichlet_alpha"], dtype=float),
                int(config["monte_carlo_trials"]),
                float(stress["continuum_sigma"]),
                int(config["seed"]) + 1000 * case_number + stress_number,
            )
            row = {
                "case": case_key,
                "stress": stress_name,
                "band_count": reference_count,
                "spacing_factor": reference_spacing,
                "center_shift_fwhm": float(stress["center_shift_fwhm"]),
                "continuum_sigma": float(stress["continuum_sigma"]),
                **metrics,
            }
            monte_rows.append(row)
            case_monte[stress_name] = metrics

        figure_payloads.append(
            {
                "label": case_config["label"],
                "grid": grid,
                "spectra": component_spectra,
                "names": component_names,
                "selected_centers": selected_centers,
                "selected_widths": selected_widths,
            }
        )
        reference_cases[case_key] = {
            "label": case_config["label"],
            "components": component_names,
            "band_count": reference_count,
            "range_um": [float(grid[0]), float(grid[-1])],
            "centers_um": selected_centers.tolist(),
            "fwhm_um": selected_widths.tolist(),
            "nominal_rmse": case_monte["nominal"]["rmse"],
            "combined_rmse": case_monte["combined"]["rmse"],
            "ga_gap_fraction": gap_fraction,
        }

    sweep = pd.DataFrame(sweep_rows)
    selected_bands = pd.DataFrame(band_rows)
    monte = pd.DataFrame(monte_rows)
    ga_frame = pd.DataFrame(ga_rows)
    sweep.to_csv(results_dir / "design_sweep.csv", index=False)
    selected_bands.to_csv(results_dir / "selected_bands.csv", index=False)
    monte.to_csv(results_dir / "unmixing_stress.csv", index=False)
    ga_frame.to_json(results_dir / "genetic_audit.json", orient="records", indent=2)
    (provenance_dir / "spectra.json").write_text(
        json.dumps(dict(sorted(provenance.items())), indent=2) + "\n"
    )

    plot_case_designs(figure_payloads, figures_dir / "three_case_designs.png")
    plot_sensitivity(sweep, figures_dir / "band_count_uncertainty.png")
    plot_spacing(sweep, figures_dir / "spacing_sensitivity.png", int(config["reference_band_count"]))
    plot_unmixing(monte, figures_dir / "unmixing_stress.png")

    summary = {
        "seed": int(config["seed"]),
        "unique_spectrum_count": len(provenance),
        "default_reference_band_count": int(config["reference_band_count"]),
        "reference_spacing_factor": float(config["reference_spacing_factor"]),
        "reference_cases": reference_cases,
        "interpretation": (
            "Laboratory-domain sensitivity study; not an at-sensor lunar radiance or "
            "flight-instrument performance prediction."
        ),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_macros(project_root / "paper" / "results.tex", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config or root / "configs" / "study.yaml"
    print(json.dumps(run(root, config), indent=2))


if __name__ == "__main__":
    main()
