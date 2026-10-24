"""Export inspectable research outputs without patient identifiers or hidden state."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import __version__
from .core import MRIStudy, QCItem, ROICurve, analyze_roi


def _finite_or_none(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def export_analysis(
    output_dir: str | Path,
    study: MRIStudy,
    mask: np.ndarray,
    curve: ROICurve,
    qc_items: list[QCItem],
) -> dict[str, Path]:
    """Write CSV, JSON, curve PNG, spatial ROI NIfTI and a self-contained HTML summary.

    ``output_dir`` should be an empty directory chosen for this export. Existing
    report filenames are rejected, so an earlier analysis cannot be overwritten.
    The HTML contains an inline PNG curve for convenient sharing; the PNG is also
    provided for slides. Input filenames and source paths are deliberately omitted.
    """
    output = Path(output_dir)
    paths = {
        key: output / name
        for key, name in {
            "curve_csv": "roi_curve.csv",
            "summary_json": "summary.json",
            "curve_png": "roi_curve.png",
            "roi_nifti": "roi_mask.nii.gz",
            "report_html": "report.html",
        }.items()
    }
    existing = [p.name for p in paths.values() if p.exists()]
    if existing:
        raise FileExistsError(
            "Choose a new export folder; existing outputs: " + ", ".join(existing)
        )
    mask_array = np.asarray(mask)
    if mask_array.shape != study.spatial_shape:
        raise ValueError("ROI mask must have the same 3D spatial shape as the study.")
    mask_array = mask_array > 0
    if not mask_array.any():
        raise ValueError("Cannot export an empty ROI.")
    if int(mask_array.sum()) != curve.voxel_count:
        raise ValueError("ROI and curve differ; recompute the analysis before exporting.")
    current = analyze_roi(study, mask_array, baseline_index=curve.baseline_index)
    for field in ("x", "mean", "std", "relative_enhancement"):
        if not np.array_equal(getattr(current, field), getattr(curve, field), equal_nan=True):
            raise ValueError("ROI and curve differ; recompute the analysis before exporting.")
    output.mkdir(parents=True, exist_ok=True)

    with paths["curve_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "phase_index",
                "time_seconds" if study.times_seconds is not None else "phase",
                "mean_signal",
                "std_signal",
                "relative_enhancement_percent",
            ]
        )
        for index, (x, mean, std, relative) in enumerate(
            zip(curve.x, curve.mean, curve.std, curve.relative_enhancement, strict=True), start=1
        ):
            writer.writerow(
                [
                    index,
                    float(x),
                    float(mean),
                    float(std),
                    float(relative) if np.isfinite(relative) else "",
                ]
            )

    summary = {
        "software": "Breast MRI Workbench",
        "version": __version__,
        "purpose": "Research and education; not a diagnostic tool.",
        "shape_tzyx": list(study.data.shape),
        "spacing_zyx": list(study.spacing_zyx),
        "spatial_unit": study.spatial_unit,
        "time_seconds": None if study.times_seconds is None else study.times_seconds.tolist(),
        "x_label": curve.x_label,
        "roi_voxel_count": curve.voxel_count,
        "baseline_index_zero_based": curve.baseline_index,
        "baseline_phase_one_based": curve.baseline_index + 1,
        "baseline_x": float(curve.x[curve.baseline_index]),
        "baseline_signal": float(curve.mean[curve.baseline_index]),
        "mean_signal": _finite_or_none(curve.mean),
        "std_signal": _finite_or_none(curve.std),
        "relative_enhancement_percent": _finite_or_none(curve.relative_enhancement),
        "relative_enhancement_definition": "100 * (mean_signal - baseline_signal) / baseline_signal",
        "curve_warnings": list(curve.warnings),
        "qc": [
            {"code": item.code, "severity": item.severity, "message": item.message}
            for item in qc_items
        ],
        "limitations": [
            "Relative signal enhancement is not contrast concentration or pharmacokinetic modeling.",
            "Quality checks are limited to observable data consistency; they do not validate diagnosis.",
            "The ROI is static across phases; motion and anatomical alignment require manual review.",
        ],
    }
    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    affine = study.affine_xyz
    if affine is None:
        affine = np.diag([*study.spacing_zyx[::-1], 1.0])
    mask_xyz = np.transpose(mask_array.astype(np.uint8), (2, 1, 0))
    roi_image = nib.Nifti1Image(mask_xyz, affine)
    roi_image.header.set_xyzt_units(xyz=study.spatial_unit)
    nib.save(roi_image, paths["roi_nifti"])

    figure = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    axes[0].plot(curve.x, curve.mean, "o-", color="#1b7f9e", linewidth=2)
    axes[0].fill_between(
        curve.x, curve.mean - curve.std, curve.mean + curve.std, alpha=0.18, color="#1b7f9e"
    )
    axes[0].set(ylabel="ROI signal (arbitrary units)", title="Mean signal ± voxel SD")
    axes[1].plot(curve.x, curve.relative_enhancement, "o-", color="#b35634", linewidth=2)
    axes[1].axhline(0, color="gray", linewidth=0.6)
    axes[1].set(ylabel="Relative enhancement (%)", title="Change from selected baseline")
    for axis in axes:
        axis.set_xlabel(curve.x_label)
        axis.grid(alpha=0.2)
    figure.savefig(paths["curve_png"], dpi=150)

    # A data URI keeps the HTML portable without external assets or network calls.
    import base64

    encoded_plot = base64.b64encode(paths["curve_png"].read_bytes()).decode("ascii")
    rows = "".join(
        f"<tr><td>{html.escape(item.severity.upper())}</td>"
        f"<td>{html.escape(item.code)}</td><td>{html.escape(item.message)}</td></tr>"
        for item in qc_items
    )
    warnings = "".join(f"<li>{html.escape(warning)}</li>" for warning in curve.warnings)
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Breast MRI Workbench — analysis report</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#193744}}
h1{{font-size:30px}}p{{line-height:1.6}}.note{{background:#edf5f7;padding:18px;border-radius:10px}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:12px;text-align:left;border-bottom:1px solid #dae5e8}}</style>
<h1>Breast MRI Workbench</h1><p>Version {__version__} · ROI analysis and data quality report</p>
<p class="note">Research and education only. No diagnosis, registration, contrast concentration
or pharmacokinetic estimates are provided. Signal units are arbitrary; the shaded band is
voxel standard deviation, not a confidence interval.</p>
<p><strong>Study shape (T, Z, Y, X):</strong> {html.escape(str(tuple(study.data.shape)))}<br>
<strong>Voxel spacing (Z, Y, X):</strong> {html.escape(str(study.spacing_zyx))} ({html.escape(study.spatial_unit)})<br>
<strong>ROI voxels:</strong> {curve.voxel_count}<br>
<strong>Baseline phase (1-based):</strong> {curve.baseline_index + 1}<br>
<strong>Time axis:</strong> {html.escape(curve.x_label)}</p>
<img alt="ROI mean signal and relative enhancement curves" src="data:image/png;base64,{encoded_plot}">
<ul>{warnings}</ul><h2>Data quality checks</h2><table><tr><th>Severity</th><th>Code</th>
<th>Observation</th></tr>{rows}</table>
<p>Machine-readable outputs: roi_curve.csv, summary.json, roi_mask.nii.gz.
Input names and paths are omitted from this report.</p></html>"""
    paths["report_html"].write_text(document, encoding="utf-8")
    return paths
