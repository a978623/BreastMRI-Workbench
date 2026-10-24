import numpy as np

from napari_breast_mri._widget import BreastMRIWidget


def test_installed_manifest_opens_widget_and_sample(make_napari_viewer):
    viewer = make_napari_viewer()
    _dock, widget = viewer.window.add_plugin_dock_widget(
        "napari-breast-mri", "Breast MRI Workbench"
    )
    assert isinstance(widget, BreastMRIWidget)
    layers = viewer.open_sample("napari-breast-mri", "synthetic-dce")
    assert len(layers) == 2
    assert layers[0].ndim == 4 and layers[1].ndim == 3
    assert layers[0].metadata["breast_mri_study"].spatial_unit == "mm"
    widget.analyze()
    assert widget.last_curve is not None
    assert np.isfinite(widget.last_curve.mean).all()
