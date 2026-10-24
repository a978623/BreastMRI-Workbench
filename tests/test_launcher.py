from unittest.mock import patch

import napari

from napari_breast_mri.__main__ import main


def test_demo_launcher_constructs_a_working_four_dimensional_viewer(qtbot):
    with patch("napari.run") as event_loop:
        try:
            assert main(["--demo"]) == 0
            event_loop.assert_called_once_with()
            viewers = list(napari.Viewer._instances)
            assert len(viewers) == 1
            viewer = viewers[0]
            assert viewer.dims.ndim == 4
            assert viewer.dims.axis_labels == ("phase", "z", "y", "x")
            assert len(viewer.layers) == 2
        finally:
            napari.Viewer.close_all()
