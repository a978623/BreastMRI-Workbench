"""Display-independent data, ROI analysis, and quality checks.

Arrays use ``(phase, z, y, x)`` throughout the package.  Signal enhancement
is a descriptive calculation, not a pharmacokinetic estimate or diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MRIStudy:
    """An MRI time series on one common physical voxel grid."""

    data: NDArray
    name: str = "DCE MRI"
    times_seconds: NDArray | None = None
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    affine_xyz: NDArray | None = None
    source_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    spatial_unit: str = "unknown"

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.data):
            raise ValueError("Complex-valued images are not supported; provide magnitude images.")
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 4 or any(size < 1 for size in self.data.shape):
            raise ValueError("Study data must be a nonempty 4D array shaped (phase, z, y, x).")
        spacing = np.asarray(self.spacing_zyx, dtype=float)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("Voxel spacing must contain three finite, strictly positive numbers.")
        self.spacing_zyx = tuple(float(value) for value in spacing)
        if self.spatial_unit not in {"unknown", "meter", "mm", "micron"}:
            raise ValueError("Spatial unit must be unknown, meter, mm, or micron.")
        if self.times_seconds is not None:
            times = np.asarray(self.times_seconds, dtype=np.float64)
            if times.ndim != 1 or len(times) != self.n_timepoints:
                raise ValueError("Provide exactly one acquisition time for every phase.")
            if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
                raise ValueError("Acquisition times must be finite and strictly increasing.")
            self.times_seconds = times.copy()
        if self.affine_xyz is not None:
            affine = np.asarray(self.affine_xyz, dtype=np.float64)
            if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                raise ValueError("The spatial affine must be a finite 4-by-4 matrix.")
            if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), rtol=0, atol=1e-8):
                raise ValueError("The spatial affine must have homogeneous last row (0, 0, 0, 1).")
            if np.linalg.matrix_rank(affine[:3, :3]) < 3:
                raise ValueError("The spatial affine must describe an invertible voxel grid.")
            affine_spacing = np.linalg.norm(affine[:3, :3], axis=0)[::-1]
            if not np.allclose(affine_spacing, spacing, rtol=1e-5, atol=1e-6):
                raise ValueError("Voxel spacing must agree with the spatial affine's voxel sizes.")
            self.affine_xyz = affine.copy()
        self.name = str(self.name) or "DCE MRI"
        self.source_files = tuple(str(path) for path in self.source_files)
        self.warnings = tuple(str(message) for message in self.warnings)

    @property
    def n_timepoints(self) -> int:
        return self.data.shape[0]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return self.data.shape[1:]


@dataclass(frozen=True)
class ROICurve:
    """Per-phase ROI statistics; standard deviation is population SD."""

    x: NDArray
    x_label: str
    mean: NDArray
    std: NDArray
    relative_enhancement: NDArray
    voxel_count: int
    warnings: tuple[str, ...] = ()
    baseline_index: int = 0


@dataclass(frozen=True)
class QCItem:
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


def analyze_roi(study: MRIStudy, mask_zyx: NDArray, baseline_index: int = 0) -> ROICurve:
    """Measure a fixed spatial ROI across every phase.

    Positive label values form the ROI. Nonfinite voxel values are excluded
    independently in each phase. A phase without any finite ROI signal is an
    error, rather than an interpolated or invented measurement.
    """
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3 or mask.shape != study.spatial_shape:
        raise ValueError(f"ROI shape must match the study spatial shape {study.spatial_shape}.")
    if np.iscomplexobj(mask):
        raise ValueError("ROI labels must be real numbers or booleans.")
    if not np.all(np.isfinite(mask)):
        raise ValueError("ROI labels must be finite.")
    selected = mask > 0
    voxel_count = int(np.count_nonzero(selected))
    if voxel_count == 0:
        raise ValueError("The ROI is empty. Paint or select at least one positive label voxel.")
    try:
        baseline = index(baseline_index)
    except TypeError as error:
        raise ValueError("Baseline phase must be an integer index.") from error
    if isinstance(baseline_index, (bool, np.bool_)) or not 0 <= baseline < study.n_timepoints:
        raise ValueError("Baseline phase is outside the time series.")

    means = np.empty(study.n_timepoints, dtype=np.float64)
    deviations = np.empty_like(means)
    excluded = 0
    for phase_index, phase in enumerate(study.data):
        values = phase[selected]
        finite = values[np.isfinite(values)]
        excluded += len(values) - len(finite)
        if len(finite) == 0:
            raise ValueError(f"Phase {phase_index + 1} contains no finite signal inside the ROI.")
        means[phase_index] = np.mean(finite, dtype=np.float64)
        deviations[phase_index] = np.std(finite, dtype=np.float64)

    messages: list[str] = []
    if excluded:
        messages.append(f"Excluded {excluded} nonfinite ROI voxel measurements across all phases.")
    baseline_mean = means[baseline]
    if abs(baseline_mean) <= 1e-8:
        enhancement = np.full_like(means, np.nan)
        messages.append(
            "Baseline ROI mean is zero or near zero; relative enhancement is undefined."
        )
    else:
        enhancement = 100.0 * (means - baseline_mean) / baseline_mean
        if baseline_mean < 0:
            messages.append(
                "Baseline ROI mean is negative; check the image signal scale before interpretation."
            )
    if study.times_seconds is None:
        x = np.arange(1, study.n_timepoints + 1, dtype=np.float64)
        x_label = "Phase"
    else:
        x = study.times_seconds.copy()
        x_label = "Time (s)"
    return ROICurve(
        x, x_label, means, deviations, enhancement, voxel_count, tuple(messages), baseline
    )


def qc_study(study: MRIStudy) -> list[QCItem]:
    """Find observable data issues without assigning clinical significance."""
    findings = [
        QCItem(
            "dimensions",
            "info",
            f"{study.n_timepoints} phase(s); spatial shape {study.spatial_shape} (z, y, x).",
        )
    ]
    if study.n_timepoints == 1:
        findings.append(
            QCItem(
                "single_timepoint",
                "warning",
                "Only one phase is present; a dynamic enhancement curve needs multiple phases.",
            )
        )
    if study.times_seconds is None:
        findings.append(
            QCItem(
                "missing_time",
                "info",
                "Physical acquisition times are unavailable. Curves use phase indices, not seconds.",
            )
        )

    nonfinite_count = 0
    all_invalid: list[int] = []
    constant: list[int] = []
    baseline_mean: float | None = None
    for phase_index, phase in enumerate(study.data):
        finite_mask = np.isfinite(phase)
        count = int(np.count_nonzero(finite_mask))
        nonfinite_count += phase.size - count
        if count == 0:
            all_invalid.append(phase_index + 1)
            continue
        finite = phase[finite_mask]
        if np.min(finite) == np.max(finite):
            constant.append(phase_index + 1)
        if phase_index == 0:
            baseline_mean = float(np.mean(finite, dtype=np.float64))
    if nonfinite_count:
        findings.append(
            QCItem(
                "nonfinite_data",
                "warning",
                f"{nonfinite_count} voxel measurements are NaN or infinite; ROI analysis excludes these measurements.",
            )
        )
    if all_invalid:
        findings.append(
            QCItem(
                "invalid_phase",
                "error",
                f"No finite signal in phase(s): {', '.join(map(str, all_invalid))}.",
            )
        )
    if constant:
        findings.append(
            QCItem(
                "constant_phase",
                "warning",
                f"Constant finite signal in phase(s): {', '.join(map(str, constant))}. Check for blank or placeholder volumes.",
            )
        )
    if baseline_mean is not None and baseline_mean <= 1e-8:
        findings.append(
            QCItem(
                "baseline_nonpositive",
                "warning",
                "The first phase has a nonpositive or near-zero whole-volume mean. Check signal scaling; ROI enhancement also depends on the selected baseline.",
            )
        )

    duplicates: list[str] = []
    for phase_index in range(1, study.n_timepoints):
        for earlier in range(phase_index):
            if np.array_equal(study.data[phase_index], study.data[earlier], equal_nan=True):
                duplicates.append(f"{phase_index + 1} = {earlier + 1}")
                break
    if duplicates:
        findings.append(
            QCItem(
                "duplicate_volumes",
                "warning",
                "Identical phase data detected (1-based phases): "
                + "; ".join(duplicates)
                + ". Verify the input series.",
            )
        )
    for warning_index, message in enumerate(study.warnings, start=1):
        findings.append(QCItem(f"source_warning_{warning_index}", "warning", message))
    return findings
