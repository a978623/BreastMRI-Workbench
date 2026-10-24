import csv
import json

import nibabel as nib
import numpy as np
import pytest

from napari_breast_mri.core import analyze_roi, qc_study
from napari_breast_mri.report import export_analysis
from napari_breast_mri.sample_data import make_demo_roi, make_demo_study


def test_report_outputs_are_inspectable_and_preserve_spatial_roi(tmp_path):
    study = make_demo_study()
    mask = make_demo_roi(study)
    curve = analyze_roi(study, mask)
    paths = export_analysis(tmp_path, study, mask, curve, qc_study(study))
    assert len(paths) == 5 and all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["roi_voxel_count"] == int(mask.sum())
    assert "source_files" not in summary
    with paths["curve_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == study.n_timepoints
    assert float(rows[0]["relative_enhancement_percent"]) == pytest.approx(0)
    restored = nib.load(paths["roi_nifti"])
    np.testing.assert_array_equal(restored.get_fdata(), np.transpose(mask, (2, 1, 0)))
    np.testing.assert_allclose(restored.affine, study.affine_xyz)
    assert restored.header.get_xyzt_units()[0] == "mm"
    assert summary["baseline_phase_one_based"] == 1
    report = paths["report_html"].read_text(encoding="utf-8")
    assert "data:image/png;base64," in report
    assert "confidence interval" in report


def test_export_refuses_to_overwrite_previous_analysis(tmp_path):
    study = make_demo_study()
    mask = make_demo_roi(study)
    curve = analyze_roi(study, mask)
    export_analysis(tmp_path, study, mask, curve, qc_study(study))
    before = (tmp_path / "summary.json").read_bytes()
    with pytest.raises(FileExistsError, match="new export folder"):
        export_analysis(tmp_path, study, mask, curve, qc_study(study))
    assert (tmp_path / "summary.json").read_bytes() == before


def test_export_rejects_a_stale_curve_after_image_edit(tmp_path):
    study = make_demo_study()
    mask = make_demo_roi(study)
    curve = analyze_roi(study, mask)
    study.data = study.data * 2
    with pytest.raises(ValueError, match="recompute"):
        export_analysis(tmp_path, study, mask, curve, qc_study(study))
    assert not list(tmp_path.iterdir())


def test_zero_reference_exports_strict_json_null_values(tmp_path):
    study = make_demo_study()
    study.data[0] = 0
    mask = make_demo_roi(study)
    curve = analyze_roi(study, mask)
    paths = export_analysis(tmp_path, study, mask, curve, qc_study(study))
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["relative_enhancement_percent"] == [None] * study.n_timepoints
