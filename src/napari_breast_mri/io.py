"""NIfTI readers with explicit array orientation and physical-grid checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from .core import MRIStudy

_TIME_UNIT_TO_SECONDS = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}


def _paths(paths: Sequence[str | Path] | str | Path) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    result = tuple(Path(path) for path in paths)
    if not result:
        raise ValueError("Select at least one NIfTI file.")
    if not all(str(path).lower().endswith((".nii", ".nii.gz")) for path in result):
        raise ValueError("Only NIfTI .nii and .nii.gz files are supported.")
    return result


def _time_axis(
    image: nib.spatialimages.SpatialImage, count: int
) -> tuple[NDArray | None, str | None]:
    unit = image.header.get_xyzt_units()[1]
    interval = float(image.header.get_zooms()[3])
    if unit in _TIME_UNIT_TO_SECONDS and np.isfinite(interval) and interval > 0:
        return np.arange(count, dtype=np.float64) * interval * _TIME_UNIT_TO_SECONDS[unit], None
    return (
        None,
        "The NIfTI temporal interval or time unit is missing/unsupported; physical acquisition times were not inferred.",
    )


def load_nifti(
    paths: Sequence[str | Path] | str | Path,
    times_seconds: NDArray | Sequence[float] | None = None,
) -> MRIStudy:
    """Load one 3D/4D file, or an explicitly ordered list of 3D files.

    Spatial axes are reoriented to closest-canonical RAS+ before conversion
    from NIfTI ``(x,y,z[,phase])`` to application ``(phase,z,y,x)``. Separate
    files must already occupy one common physical grid; no registration or
    implicit resampling is performed. Filename order never implies time.
    """
    selected = _paths(paths)
    images = []
    for path in selected:
        source = nib.load(str(path))
        if np.dtype(source.get_data_dtype()).kind == "c":
            raise ValueError(
                f"Complex-valued images are not supported ({path.name}); provide magnitude images."
            )
        images.append(nib.as_closest_canonical(source))
    if any(len(image.shape) not in (3, 4) for image in images):
        raise ValueError("Each NIfTI image must be a 3D volume or a 4D time series.")
    if len(images) > 1 and any(len(image.shape) != 3 for image in images):
        raise ValueError(
            "When selecting multiple files, each file must contain a 3D volume; do not mix 3D and 4D files."
        )
    reference = images[0]
    reference_unit = reference.header.get_xyzt_units()[0]
    for path, image in zip(selected[1:], images[1:]):
        same_shape = image.shape[:3] == reference.shape[:3]
        same_affine = np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4)
        same_unit = image.header.get_xyzt_units()[0] == reference_unit
        if not (same_shape and same_affine and same_unit):
            raise ValueError(
                f"Incompatible physical grid in {path.name}. Align/resample the volumes explicitly before stacking them."
            )

    messages: list[str] = []
    if reference_unit == "unknown":
        messages.append(
            "NIfTI spatial units are unknown; voxel spacing retains the file's native units."
        )
    elif reference_unit != "mm":
        messages.append(
            f"Voxel spacing is expressed in the NIfTI file's native spatial unit: {reference_unit}."
        )
    if len(images) > 1:
        arrays = [
            np.asarray(image.get_fdata(dtype=np.float32)).transpose(2, 1, 0) for image in images
        ]
        data = np.stack(arrays, axis=0)
        messages.append(
            "Separate 3D files were stacked in the supplied order. Verify this order against acquisition metadata; filenames are not interpreted as times."
        )
    else:
        array = reference.get_fdata(dtype=np.float32)
        if array.ndim == 3:
            data = array.transpose(2, 1, 0)[np.newaxis, ...]
        else:
            data = array.transpose(3, 2, 1, 0)
            if times_seconds is None:
                times_seconds, warning = _time_axis(reference, data.shape[0])
                if warning:
                    messages.append(warning)
    spatial_spacing = nib.affines.voxel_sizes(reference.affine)
    name = selected[0].name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if len(selected) > 1:
        name = f"{name} ({len(selected)} phases)"
    return MRIStudy(
        data=np.ascontiguousarray(data, dtype=np.float32),
        name=name,
        times_seconds=times_seconds,
        spacing_zyx=tuple(float(value) for value in spatial_spacing[::-1]),
        affine_xyz=reference.affine,
        source_files=tuple(str(path) for path in selected),
        warnings=tuple(messages),
        spatial_unit=reference_unit,
    )


def study_layer_data(study: MRIStudy) -> tuple[NDArray, dict[str, Any], str]:
    """Build a napari image layer while preserving study metadata."""
    return (
        study.data,
        {
            "name": study.name,
            "scale": (1.0, *study.spacing_zyx),
            "axis_labels": ("phase", "z", "y", "x"),
            "metadata": {"breast_mri_study": study},
            "rgb": False,
        },
        "image",
    )


def get_reader(path: str | list[str]) -> Callable | None:
    """napari reader hook: unknown file types are left to other readers."""
    try:
        _paths(path)
    except (TypeError, ValueError):
        return None

    def reader(paths: str | list[str]):
        return [study_layer_data(load_nifti(paths))]

    return reader
