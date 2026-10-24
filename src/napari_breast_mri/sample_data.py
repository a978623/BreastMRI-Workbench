"""Small deterministic synthetic phantoms, with no patient information."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .core import MRIStudy
from .io import study_layer_data

_DEMO_SHAPE = (12, 48, 48)


def make_demo_roi(study: MRIStudy | None = None) -> NDArray:
    """Return a positive demonstration ROI on the requested spatial grid."""
    shape = _DEMO_SHAPE if study is None else study.spatial_shape
    z, y, x = np.indices(shape, dtype=float)
    center = ((shape[0] - 1) * 0.5, (shape[1] - 1) * 0.47, (shape[2] - 1) * 0.69)
    radii = tuple(max(0.5, size * fraction) for size, fraction in zip(shape, (0.22, 0.115, 0.115)))
    distance = sum(
        ((axis - location) / radius) ** 2
        for axis, location, radius in zip((z, y, x), center, radii)
    )
    mask = distance <= 1.0
    if not np.any(mask):
        mask[tuple(round(value) for value in center)] = True
    return mask


def make_demo_study(seed: int = 7) -> MRIStudy:
    """Generate a bilateral phantom with a deliberately changing ROI signal."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(_DEMO_SHAPE, dtype=float)
    zc, yc = (_DEMO_SHAPE[0] - 1) / 2, (_DEMO_SHAPE[1] - 1) / 2
    left = ((z - zc) / 5.0) ** 2 + ((y - yc) / 19.0) ** 2 + ((x - 13.0) / 10.0) ** 2 <= 1
    right = ((z - zc) / 5.0) ** 2 + ((y - yc) / 19.0) ** 2 + ((x - 34.0) / 10.0) ** 2 <= 1
    tissue = left | right
    baseline = 15.0 + 65.0 * tissue + 8.0 * np.sin(x / 8.0) * tissue
    roi = make_demo_roi()
    enhancement = np.asarray([0.0, 0.25, 0.80, 0.65, 0.45, 0.30])
    phases = []
    for phase_index, change in enumerate(enhancement):
        phase = baseline + 8.0 * (1 - np.exp(-phase_index / 2.0)) * tissue
        phase = phase + 95.0 * change * roi + rng.normal(0, 1.3, _DEMO_SHAPE)
        phases.append(phase.astype(np.float32))
    return MRIStudy(
        data=np.stack(phases),
        name="Synthetic breast DCE-MRI (not patient data)",
        times_seconds=np.arange(len(phases), dtype=float) * 60.0,
        spacing_zyx=(2.0, 1.0, 1.0),
        affine_xyz=np.diag([1.0, 1.0, 2.0, 1.0]),
        spatial_unit="mm",
    )


def make_sample_data():
    """napari sample-data hook, including an editable positive-label ROI."""
    study = make_demo_study()
    image = study_layer_data(study)
    image[1]["metadata"]["synthetic"] = True
    labels = (
        make_demo_roi(study).astype(np.uint8),
        {
            "name": "Demo ROI",
            "scale": study.spacing_zyx,
            "axis_labels": ("z", "y", "x"),
            "metadata": {"synthetic": True},
        },
        "labels",
    )
    return [image, labels]
