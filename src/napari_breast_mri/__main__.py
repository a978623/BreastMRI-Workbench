"""Launch the viewer or export the deterministic demo without opening a GUI."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Breast MRI Workbench for napari")
    parser.add_argument("--demo", action="store_true", help="Open a synthetic DCE study and ROI")
    parser.add_argument(
        "--export-demo",
        type=Path,
        metavar="FOLDER",
        help="Export a synthetic example report without opening napari",
    )
    args = parser.parse_args(argv)
    if args.export_demo is not None:
        from .core import analyze_roi, qc_study
        from .report import export_analysis
        from .sample_data import make_demo_roi, make_demo_study

        study = make_demo_study()
        mask = make_demo_roi(study)
        paths = export_analysis(
            args.export_demo, study, mask, analyze_roi(study, mask), qc_study(study)
        )
        for key, path in paths.items():
            print(f"{key}: {path}")
        return 0

    import napari

    from ._widget import BreastMRIWidget

    viewer = napari.Viewer(title="Breast MRI Workbench", axis_labels=("phase", "z", "y", "x"))
    widget = BreastMRIWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Breast MRI Workbench", area="right")
    if args.demo:
        widget.load_demo()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
