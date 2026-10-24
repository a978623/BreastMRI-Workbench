import numpy as np
import pytest

from napari_breast_mri.core import MRIStudy, analyze_roi, qc_study


def roi_fixture():
    baseline = np.full((2, 2, 2), 99.0)
    baseline[0, 0, :] = [2.0, 4.0]
    data = np.stack([baseline, baseline * 2, baseline / 2])
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, :] = 3
    return MRIStudy(data, times_seconds=np.array([0.0, 30.0, 90.0])), mask


def test_roi_statistics_use_only_positive_labels_and_physical_time():
    study, mask = roi_fixture()
    curve = analyze_roi(study, mask)
    np.testing.assert_allclose(curve.mean, [3.0, 6.0, 1.5])
    np.testing.assert_allclose(curve.std, [1.0, 2.0, 0.5])
    np.testing.assert_allclose(curve.relative_enhancement, [0.0, 100.0, -50.0])
    np.testing.assert_array_equal(curve.x, [0.0, 30.0, 90.0])
    assert curve.voxel_count == 2
    assert curve.x_label == "Time (s)"


def test_selected_baseline_changes_relative_enhancement():
    study, mask = roi_fixture()
    curve = analyze_roi(study, mask, baseline_index=1)
    np.testing.assert_allclose(curve.relative_enhancement, [-50.0, 0.0, -75.0])
    assert curve.baseline_index == 1
    assert curve.mean[curve.baseline_index] == 6.0
    assert curve.x[curve.baseline_index] == 30.0


def test_missing_time_uses_one_based_phase_axis():
    study, mask = roi_fixture()
    study.times_seconds = None
    curve = analyze_roi(study, mask)
    np.testing.assert_array_equal(curve.x, [1.0, 2.0, 3.0])
    assert curve.x_label == "Phase"


def test_nonfinite_roi_measurements_are_excluded_but_empty_phase_is_rejected():
    study, mask = roi_fixture()
    study.data[1, 0, 0, 0] = np.nan
    curve = analyze_roi(study, mask)
    assert curve.mean[1] == 8.0
    assert curve.std[1] == 0.0
    assert "Excluded 1" in curve.warnings[0]
    study.data[1, 0, 0, 1] = np.inf
    with pytest.raises(ValueError, match="Phase 2 contains no finite"):
        analyze_roi(study, mask)


def test_zero_baseline_yields_undefined_enhancement_without_division_warning():
    study = MRIStudy(np.stack([np.zeros((2, 2, 2)), np.ones((2, 2, 2))]))
    with np.errstate(all="raise"):
        curve = analyze_roi(study, np.ones((2, 2, 2)))
    assert np.all(np.isnan(curve.relative_enhancement))
    assert "undefined" in curve.warnings[0]


@pytest.mark.parametrize(
    "mask",
    [np.zeros((2, 2, 2)), np.ones((3, 2, 2)), np.ones((1, 2, 2, 2)), np.full((2, 2, 2), np.nan)],
)
def test_unusable_roi_is_rejected(mask):
    study, _ = roi_fixture()
    with pytest.raises(ValueError):
        analyze_roi(study, mask)


@pytest.mark.parametrize("baseline", [-1, 3, 1.5, True])
def test_invalid_baseline_index_is_rejected(baseline):
    study, mask = roi_fixture()
    with pytest.raises(ValueError):
        analyze_roi(study, mask, baseline)


@pytest.mark.parametrize("times", [[0, 1], [0, 0, 1], [0, np.nan, 1], [0, 2, 1]])
def test_invalid_acquisition_time_axis_is_rejected(times):
    with pytest.raises(ValueError):
        MRIStudy(np.ones((3, 2, 2, 2)), times_seconds=times)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": np.ones((2, 2, 2))},
        {"data": np.ones((0, 2, 2, 2))},
        {"data": np.ones((1, 2, 2, 2)), "spacing_zyx": (1, 0, 1)},
        {"data": np.ones((1, 2, 2, 2)), "affine_xyz": np.zeros((4, 4))},
    ],
)
def test_invalid_study_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MRIStudy(**kwargs)


def test_affine_last_row_and_spacing_consistency_are_validated():
    invalid = np.eye(4)
    invalid[3, 0] = 0.1
    with pytest.raises(ValueError, match="homogeneous last row"):
        MRIStudy(np.ones((1, 2, 2, 2)), affine_xyz=invalid)
    with pytest.raises(ValueError, match="agree with the spatial affine"):
        MRIStudy(np.ones((1, 2, 2, 2)), affine_xyz=np.diag([2, 1, 1, 1]))
    with pytest.raises(ValueError, match="Spatial unit"):
        MRIStudy(np.ones((1, 2, 2, 2)), spatial_unit="cm")


def test_qc_finds_duplicate_nan_volumes_and_completely_invalid_phase():
    first = np.arange(8, dtype=float).reshape(2, 2, 2)
    first[0, 0, 0] = np.nan
    study = MRIStudy(np.stack([first, first.copy(), np.full_like(first, np.nan)]))
    findings = {item.code: item for item in qc_study(study)}
    assert findings["nonfinite_data"].severity == "warning"
    assert findings["invalid_phase"].severity == "error"
    assert "2 = 1" in findings["duplicate_volumes"].message
    assert "missing_time" in findings


def test_qc_finds_single_blank_baseline_and_propagates_source_warning():
    study = MRIStudy(np.zeros((1, 2, 2, 2)), warnings=("Check supplied file order.",))
    findings = {item.code: item for item in qc_study(study)}
    assert {"single_timepoint", "constant_phase", "baseline_nonpositive"} <= findings.keys()
    assert findings["source_warning_1"].message == "Check supplied file order."
