"""Capture the real napari demo window for the README (requires a Qt/OpenGL display)."""

from pathlib import Path

import napari
from qtpy.QtCore import QTimer

from napari_breast_mri._widget import BreastMRIWidget


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    viewer = napari.Viewer(
        axis_labels=("phase", "z", "y", "x"),
        title="Breast MRI Workbench · synthetic demo",
        show=False,
    )
    viewer.theme = "light"
    widget = BreastMRIWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Breast MRI Workbench", area="right")
    widget.load_demo()
    if not widget._analysis_ready:
        raise RuntimeError(widget.status_label.text())
    viewer.dims.set_current_step(0, 2)
    viewer.dims.set_current_step(1, 6)
    viewer.layers.selection.active = widget.image_combo.currentData()
    viewer.window.resize(1580, 1120)
    viewer.reset_view()
    errors = []

    def capture() -> None:
        try:
            viewer.screenshot(path=str(output / "workbench.png"), canvas_only=False, flash=False)
            widget.figure.savefig(output / "roi-curves.png", dpi=150, bbox_inches="tight")
            print(f"Captured {output / 'workbench.png'}")
        except Exception as error:
            errors.append(error)
        finally:
            viewer.close()

    viewer.show()
    viewer.window.resize(1580, 1120)
    QTimer.singleShot(1800, capture)
    napari.run()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
