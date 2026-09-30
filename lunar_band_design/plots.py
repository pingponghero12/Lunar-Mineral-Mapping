"""Publication figures generated only from machine-readable study outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_case_designs(case_payloads: list[dict], destination: Path) -> None:
    columns = 2 if len(case_payloads) >= 4 else 1
    rows = int(np.ceil(len(case_payloads) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.5, 3.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for axis, payload in zip(axes, case_payloads, strict=True):
        grid = payload["grid"]
        spectra = payload["spectra"]
        for index, name in enumerate(payload["names"]):
            label = name if len(payload["names"]) <= 8 else "_nolegend_"
            axis.plot(grid, spectra[:, index], lw=1.0, color=colors[index % 10], label=label)
        for band_index, (center, width) in enumerate(
            zip(payload["selected_centers"], payload["selected_widths"], strict=True)
        ):
            axis.axvspan(
                center - width / 2,
                center + width / 2,
                color="crimson",
                alpha=0.18,
                label="selected response" if band_index == 0 else None,
            )
        axis.axvline(3.0, color="0.5", ls=":", lw=0.8)
        axis.axvline(20.0, color="0.5", ls=":", lw=0.8)
        axis.set_title(payload["label"], loc="left", fontweight="bold")
        axis.set_ylabel("Laboratory reflectance")
        axis.grid(alpha=0.18)
        if len(payload["names"]) > 8:
            axis.text(0.02, 0.95, f"{len(payload['names'])} separate mineral spectra",
                      transform=axis.transAxes, va="top", fontsize=7)
        axis.legend(ncol=min(4, len(payload["names"]) + 1), fontsize=6.5, loc="best")
        axis.set_xlabel("Wavelength (µm)")
    for axis in axes[len(case_payloads):]:
        axis.set_visible(False)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_sensitivity(sweep: pd.DataFrame, destination: Path) -> None:
    cases = list(sweep["case"].drop_duplicates())
    columns = 2 if len(cases) >= 4 else len(cases)
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.5, 3.2 * rows),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, case in zip(axes, cases, strict=True):
        subset = sweep[(sweep.case == case) & (sweep.spacing_factor == 1.0)]
        for profile, group in subset.groupby("noise_profile"):
            axis.plot(group.band_count, group.mean_crlb, marker="o", label=profile)
        axis.set_title(case.replace("_", " ").title(), fontsize=9)
        axis.set_xlabel("Number of bands")
        axis.grid(alpha=0.25)
    for axis in axes[::columns]:
        axis.set_ylabel("Mean linearised uncertainty")
    axes[len(cases) - 1].legend(fontsize=7)
    for axis in axes[len(cases):]:
        axis.set_visible(False)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_spacing(sweep: pd.DataFrame, destination: Path, band_count: int) -> None:
    subset = sweep[(sweep.band_count == band_count) & (sweep.noise_profile == "reference")]
    fig, axis = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    for case, group in subset.groupby("case"):
        axis.plot(group.spacing_factor, group.objective, marker="o", label=case.replace("_", " "))
    axis.set_xlabel("Minimum spacing (sum of response half-widths × factor)")
    axis.set_ylabel("Regularised log-determinant objective")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def plot_unmixing(results: pd.DataFrame, destination: Path) -> None:
    cases = list(results["case"].drop_duplicates())
    x = np.arange(len(cases))
    width = 0.24
    fig, axis = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    for offset, stress in enumerate(["nominal", "calibration", "combined"]):
        values = [
            float(results[(results.case == case) & (results.stress == stress)].rmse.iloc[0])
            for case in cases
        ]
        axis.bar(x + (offset - 1) * width, np.array(values) * 100, width, label=stress)
    axis.set_xticks(x, [case.replace("_", "\n") for case in cases])
    axis.set_ylabel("Abundance RMSE (percentage points)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(destination, dpi=220)
    plt.close(fig)
