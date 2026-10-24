"""Behavioral tests for the napari widget using the synthetic data workflow."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from napari_breast_mri._widget import BreastMRIWidget, _PhaseOrderDialog


@pytest.fixture
def workbench(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = BreastMRIWidget(viewer)
    qtbot.addWidget(widget)
    return widget


def test_demo_has_static_roi_and_measured_seconds(workbench):
    workbench.load_demo()
    assert workbench._analysis_ready
    assert workbench.last_curve is not None
    assert workbench.last_curve.voxel_count > 0
    assert workbench.last_curve.x_label.lower().startswith("time")
    assert workbench.image_combo.currentData().ndim == 4
    assert workbench.roi_combo.currentData().ndim == 3
    assert "Synthetic demo ready" in workbench.status_label.text()


def test_existing_layer_uses_phase_indices_without_inventing_seconds(workbench):
    image = workbench.viewer.add_image(
        np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6), name="Existing", rgb=False
    )
    roi = workbench.viewer.add_labels(np.ones((4, 5, 6), dtype=np.uint8), name="Mask")
    workbench.image_combo.setCurrentIndex(workbench.image_combo.findData(image))
    workbench.roi_combo.setCurrentIndex(workbench.roi_combo.findData(roi))
    study, _mask, curve, _qc = workbench._compute_current()
    assert study.times_seconds is None
    assert "phase" in curve.x_label.lower()
    np.testing.assert_array_equal(curve.x, [1, 2, 3])
    workbench.times_input.setText("0, 55, 118")
    timed_study, _mask, timed_curve, _qc = workbench._compute_current()
    np.testing.assert_array_equal(timed_study.times_seconds, [0, 55, 118])
    assert timed_curve.x_label.lower().startswith("time")


def test_empty_roi_is_a_visible_actionable_error(workbench):
    workbench.load_demo()
    roi = workbench.roi_combo.currentData()
    roi.data = np.zeros_like(roi.data)
    workbench.analyze()
    assert not workbench._analysis_ready
    assert "empty" in workbench.status_label.text().lower()
    assert workbench.last_curve is None


def test_same_shape_but_wrong_spatial_transform_is_rejected(workbench):
    workbench.load_demo()
    roi = workbench.roi_combo.currentData()
    roi.translate = (5.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="spatial transforms differ"):
        workbench._compute_current()


def test_rotated_roi_is_rejected_even_with_matching_scale_and_translate(workbench):
    workbench.load_demo()
    roi = workbench.roi_combo.currentData()
    transform = np.eye(4)
    transform[1:3, 1:3] = [[0.0, -1.0], [1.0, 0.0]]
    roi.affine = transform
    with pytest.raises(ValueError, match="spatial transforms differ"):
        workbench._compute_current()


def test_mri_with_four_x_voxels_is_not_interpreted_as_rgba(workbench):
    from napari_breast_mri.core import MRIStudy

    image = workbench._add_study(MRIStudy(np.ones((2, 3, 5, 4))))
    assert image.ndim == 4
    assert not image.rgb


def test_spatial_transform_must_not_change_the_world_time_coordinate(workbench):
    workbench.load_demo()
    image = workbench.image_combo.currentData()
    transform = np.eye(5)
    transform[0, 1] = 0.25
    image.affine = transform
    with pytest.raises(ValueError, match="mixes space and time"):
        workbench._compute_current()


def test_create_roi_preserves_rotated_spatial_grid(workbench):
    image = workbench.viewer.add_image(np.ones((2, 4, 5, 6)), name="Transformed", rgb=False)
    transform = np.eye(5)
    transform[1:4, -1] = (11, 22, 33)
    transform[1:4, 1:4] = [[2, 0, 0], [0, 0, -3], [0, 4, 0]]
    image.affine = transform
    workbench.image_combo.setCurrentIndex(workbench.image_combo.findData(image))
    workbench.create_roi()
    roi = workbench.roi_combo.currentData()
    assert roi is not None
    roi.data = np.ones_like(roi.data)
    _study, mask, curve, _qc = workbench._compute_current()
    assert mask.all()
    assert curve.voxel_count == 4 * 5 * 6
    np.testing.assert_allclose(_study.affine_xyz[:3, 3], [33.0, 22.0, 11.0])
    assert "unknown spatial units" in _study.warnings[0]


def test_edited_image_is_used_on_export(workbench, tmp_path, monkeypatch):
    import napari_breast_mri._widget as module

    workbench.load_demo()
    before = workbench.last_curve.mean.copy()
    image = workbench.image_combo.currentData()
    image.data = np.asarray(image.data) * 2.0
    monkeypatch.setattr(module.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    received = {}

    def capture_export(destination, study, mask, curve, items):
        received["destination"] = Path(destination)
        received["curve"] = curve
        return {"csv": Path(destination) / "curve.csv"}

    monkeypatch.setattr(module, "export_analysis", capture_export)
    workbench.export_report()
    np.testing.assert_allclose(received["curve"].mean, before * 2.0)
    assert received["destination"].parent == tmp_path
    assert workbench.last_export
    assert "Exported 1 files" in workbench.status_label.text()


def test_phase_order_can_be_explicitly_rearranged(qtbot):
    dialog = _PhaseOrderDialog(["post.nii.gz", "pre.nii.gz"], None)
    qtbot.addWidget(dialog)
    dialog.files.setCurrentRow(1)
    dialog._move(-1)
    assert dialog.paths == ["pre.nii.gz", "post.nii.gz"]


def test_painting_invalidates_the_displayed_measurement(workbench):
    workbench.load_demo()
    roi = workbench.roi_combo.currentData()
    roi.brush_size = 1
    roi.paint((0, 0, 0), new_label=1)
    assert workbench.last_curve is None
    assert not workbench._analysis_ready


def test_export_uses_the_live_roi_instead_of_a_cached_mask(workbench, tmp_path, monkeypatch):
    import napari_breast_mri._widget as module

    workbench.load_demo()
    roi = workbench.roi_combo.currentData()
    original_count = workbench.last_curve.voxel_count
    smaller_mask = np.asarray(roi.data).copy()
    first_voxel = np.argwhere(smaller_mask == 1)[0]
    smaller_mask[tuple(first_voxel)] = 0
    roi.data = smaller_mask
    monkeypatch.setattr(module.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    received = {}

    def capture_export(destination, study, mask, curve, items):
        received["count"] = int(mask.sum())
        received["curve_count"] = curve.voxel_count
        return {"csv": Path(destination) / "curve.csv"}

    monkeypatch.setattr(module, "export_analysis", capture_export)
    workbench.export_report()
    assert received["count"] == original_count - 1
    assert received["curve_count"] == original_count - 1


def test_removing_selected_layer_clears_measurement(workbench):
    workbench.load_demo()
    workbench.viewer.layers.remove(workbench.roi_combo.currentData())
    assert workbench.roi_combo.count() == 0
    assert workbench.last_curve is None
    workbench.analyze()
    assert "ROI" in workbench.status_label.text()
