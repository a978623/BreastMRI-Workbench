import nibabel as nib
import numpy as np
import pytest

from napari_breast_mri.io import get_reader, load_nifti


def save_image(path, array, affine=None, time_unit="unknown", interval=1.0, space_unit="mm"):
    image = nib.Nifti1Image(
        np.asarray(array, dtype=np.float32), np.eye(4) if affine is None else affine
    )
    image.header.set_xyzt_units(space_unit, time_unit)
    if array.ndim == 4:
        image.header.set_zooms((*image.header.get_zooms()[:3], interval))
    nib.save(image, path)
    return path


def test_4d_nifti_axis_order_spacing_and_physical_time(tmp_path):
    array = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
    path = save_image(tmp_path / "dynamic.nii.gz", array, np.diag([0.7, 0.8, 2.0, 1.0]), "sec", 5.0)
    study = load_nifti([path])
    assert study.data.shape == (3, 4, 3, 2)
    assert study.data[1, 3, 2, 1] == array[1, 2, 3, 1]
    np.testing.assert_allclose(study.spacing_zyx, [2.0, 0.8, 0.7])
    np.testing.assert_allclose(study.times_seconds, [0.0, 5.0, 10.0])
    assert study.name == "dynamic"
    assert study.spatial_unit == "mm"


@pytest.mark.parametrize("unit,interval", [("sec", 30.0), ("msec", 30000.0), ("usec", 30000000.0)])
def test_temporal_units_are_converted_to_seconds(tmp_path, unit, interval):
    path = save_image(
        tmp_path / "time.nii", np.ones((2, 3, 4, 3)), time_unit=unit, interval=interval
    )
    np.testing.assert_allclose(load_nifti(path).times_seconds, [0, 30, 60])


def test_unknown_temporal_units_do_not_invent_seconds(tmp_path):
    path = save_image(tmp_path / "unknown.nii", np.ones((2, 3, 4, 3)), interval=15.0)
    study = load_nifti(path)
    assert study.times_seconds is None
    assert any("temporal" in warning for warning in study.warnings)


def test_canonical_reorientation_preserves_values_and_physical_grid(tmp_path):
    array = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    affine[0, 3] = 6.0
    path = save_image(tmp_path / "flipped.nii.gz", array, affine)
    study = load_nifti(path)
    np.testing.assert_array_equal(study.data[0], array[::-1, :, :].transpose(2, 1, 0))
    assert nib.aff2axcodes(study.affine_xyz) == ("R", "A", "S")
    np.testing.assert_allclose(study.spacing_zyx, (4, 3, 2))


def test_multifile_loading_preserves_explicit_order(tmp_path):
    late = save_image(tmp_path / "z_late.nii", np.full((2, 3, 4), 8.0))
    early = save_image(tmp_path / "a_early.nii", np.full((2, 3, 4), 2.0))
    study = load_nifti([late, early])
    np.testing.assert_array_equal(study.data[:, 0, 0, 0], [8, 2])
    assert study.times_seconds is None
    assert any("supplied order" in warning for warning in study.warnings)


def test_mismatched_physical_grid_is_rejected(tmp_path):
    first = save_image(tmp_path / "first.nii", np.ones((2, 3, 4)))
    translated = np.eye(4)
    translated[0, 3] = 5.0
    second = save_image(tmp_path / "second.nii", np.ones((2, 3, 4)), translated)
    with pytest.raises(ValueError, match="physical grid"):
        load_nifti([first, second])


def test_mismatched_spatial_units_are_rejected(tmp_path):
    first = save_image(tmp_path / "first.nii", np.ones((2, 3, 4)), space_unit="mm")
    second = save_image(tmp_path / "second.nii", np.ones((2, 3, 4)), space_unit="meter")
    with pytest.raises(ValueError, match="physical grid"):
        load_nifti([first, second])


@pytest.mark.parametrize("unit", ["unknown", "mm", "meter", "micron"])
def test_native_spatial_unit_is_preserved_without_rescaling(tmp_path, unit):
    path = save_image(
        tmp_path / "units.nii", np.ones((2, 3, 4)), np.diag([2, 3, 4, 1]), space_unit=unit
    )
    study = load_nifti(path)
    assert study.spatial_unit == unit
    np.testing.assert_allclose(study.spacing_zyx, [4, 3, 2])
    np.testing.assert_allclose(study.affine_xyz, np.diag([2, 3, 4, 1]))


def test_multifile_4d_and_unsupported_dimensions_are_rejected(tmp_path):
    three = save_image(tmp_path / "three.nii", np.ones((2, 3, 4)))
    four = save_image(tmp_path / "four.nii", np.ones((2, 3, 4, 2)))
    with pytest.raises(ValueError, match="each file must contain a 3D"):
        load_nifti([three, four])
    two = save_image(tmp_path / "two.nii", np.ones((2, 3)))
    with pytest.raises(ValueError, match="3D volume or a 4D"):
        load_nifti(two)


def test_complex_nifti_is_rejected_before_float_conversion(tmp_path):
    path = tmp_path / "complex.nii.gz"
    image = nib.Nifti1Image(np.full((2, 3, 4), 2 + 5j, dtype=np.complex64), np.eye(4))
    nib.save(image, path)
    with pytest.raises(ValueError, match="Complex-valued images are not supported"):
        load_nifti(path)


def test_explicit_nonuniform_times_override_header(tmp_path):
    path = save_image(tmp_path / "time.nii", np.ones((2, 3, 4, 3)), time_unit="sec", interval=5.0)
    np.testing.assert_array_equal(
        load_nifti(path, times_seconds=[0, 8, 17]).times_seconds, [0, 8, 17]
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_nifti(path, times_seconds=[0, 8])


def test_reader_declines_other_formats_and_retains_study_metadata(tmp_path):
    assert get_reader("scan.dcm") is None
    assert get_reader([]) is None
    path = save_image(tmp_path / "reader.nii.gz", np.ones((2, 3, 4)))
    reader = get_reader(str(path))
    data, properties, layer_type = reader(str(path))[0]
    assert layer_type == "image"
    assert data.shape == (1, 4, 3, 2)
    assert properties["metadata"]["breast_mri_study"].data is data
    assert properties["axis_labels"] == ("phase", "z", "y", "x")
    assert properties["rgb"] is False


def test_reader_does_not_treat_three_voxel_wide_time_series_as_rgb(tmp_path):
    path = save_image(tmp_path / "narrow.nii", np.ones((3, 4, 5, 2)))
    data, properties, _ = get_reader(str(path))(str(path))[0]
    assert data.shape == (2, 5, 4, 3)
    assert properties["rgb"] is False
