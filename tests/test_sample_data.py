import numpy as np

from napari_breast_mri.core import analyze_roi, qc_study
from napari_breast_mri.sample_data import make_demo_roi, make_demo_study, make_sample_data


def test_demo_is_deterministic_and_contains_a_dynamic_roi():
    first = make_demo_study()
    np.testing.assert_array_equal(first.data, make_demo_study().data)
    assert not np.array_equal(first.data, make_demo_study(seed=8).data)
    mask = make_demo_roi(first)
    assert mask.dtype == bool and mask.shape == first.spatial_shape
    assert np.any(mask)
    curve = analyze_roi(first, mask)
    assert np.argmax(curve.mean) == 2
    assert curve.relative_enhancement[2] > 50
    assert not any(item.severity == "error" for item in qc_study(first))
    assert first.spatial_unit == "mm"


def test_sample_image_and_roi_layers_share_spatial_geometry():
    image, labels = make_sample_data()
    assert image[2] == "image" and labels[2] == "labels"
    assert image[0].shape[1:] == labels[0].shape
    assert image[1]["scale"][1:] == labels[1]["scale"]
    assert labels[1]["name"] == "Demo ROI"
    assert image[1]["metadata"]["synthetic"] is True
