"""Interactive napari controls for a static ROI in a dynamic MRI study."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import napari
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.layers import Image, Labels
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .core import MRIStudy, analyze_roi, qc_study
from .io import load_nifti
from .report import export_analysis
from .sample_data import make_demo_roi, make_demo_study

if TYPE_CHECKING:
    from .core import QCItem, ROICurve


class _PhaseOrderDialog(QDialog):
    """Make the order of separately acquired volumes an explicit user choice."""

    def __init__(self, paths: list[str], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Confirm acquisition order")
        layout = QVBoxLayout(self)
        message = QLabel(
            "The first file becomes phase 1. Arrange these volumes in acquisition "
            "order; filenames do not establish acquisition times."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        self.files = QListWidget()
        self.files.addItems(paths)
        self.files.setCurrentRow(0)
        layout.addWidget(self.files)
        controls = QHBoxLayout()
        up = QPushButton("Move up")
        down = QPushButton("Move down")
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        controls.addWidget(up)
        controls.addWidget(down)
        layout.addLayout(controls)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 300)

    def _move(self, offset: int) -> None:
        current = self.files.currentRow()
        destination = current + offset
        if current < 0 or not 0 <= destination < self.files.count():
            return
        item = self.files.takeItem(current)
        self.files.insertItem(destination, item)
        self.files.setCurrentRow(destination)

    @property
    def paths(self) -> list[str]:
        return [self.files.item(i).text() for i in range(self.files.count())]


class BreastMRIWidget(QWidget):
    """A CPU-first workbench for NIfTI data and a painted 3D Labels ROI.

    Existing napari images must use ZYX (3D) or TZYX (4D) axes. Images read by
    this plugin are already converted to that convention. A single static ROI
    is measured across all phases. There is no automatic registration or
    diagnostic interpretation.
    """

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._connected_emitters: list[tuple[object, object]] = []
        self._viewer_events_connected = True
        self._analysis_ready = False
        self.last_curve: ROICurve | None = None
        self.last_export: dict[str, Path] | None = None
        self.setMinimumWidth(370)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)
        title = QLabel("BreastMRI Workbench")
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 3)
        title.setFont(font)
        layout.addWidget(title)
        subtitle = QLabel("Explore dynamic MRI signal. Research use only.")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        data_group = QGroupBox("1  Choose a study")
        data_layout = QVBoxLayout(data_group)
        open_row = QHBoxLayout()
        self.open_button = QPushButton("Open NIfTI…")
        self.demo_button = QPushButton("Load synthetic demo")
        self.open_button.clicked.connect(self.open_nifti)
        self.demo_button.clicked.connect(self.load_demo)
        open_row.addWidget(self.open_button)
        open_row.addWidget(self.demo_button)
        data_layout.addLayout(open_row)
        data_form = QFormLayout()
        self.image_combo = QComboBox()
        self.image_combo.setToolTip("3D ZYX or 4D TZYX Image layers.")
        self.times_input = QLineEdit()
        self.times_input.setPlaceholderText("Optional seconds: 0, 60, 120")
        self.times_input.setToolTip(
            "Enter one measured acquisition time in seconds per phase. Leave "
            "empty to use verified NIfTI timing or phase indices."
        )
        self.baseline_spin = QSpinBox()
        self.baseline_spin.setMinimum(1)
        self.baseline_spin.setToolTip(
            "The chosen reference phase must be appropriate for your study. "
            "This widget does not infer the contrast injection time."
        )
        data_form.addRow("Image", self.image_combo)
        data_form.addRow("Acquisition times (s)", self.times_input)
        data_form.addRow("Reference phase", self.baseline_spin)
        data_layout.addLayout(data_form)
        layout.addWidget(data_group)

        roi_group = QGroupBox("2  Select or paint a 3D ROI")
        roi_layout = QVBoxLayout(roi_group)
        roi_form = QFormLayout()
        self.roi_combo = QComboBox()
        self.label_spin = QSpinBox()
        self.label_spin.setRange(1, 2_147_483_647)
        roi_form.addRow("Labels layer", self.roi_combo)
        roi_form.addRow("Label ID", self.label_spin)
        roi_layout.addLayout(roi_form)
        self.new_roi_button = QPushButton("Create empty ROI labels")
        self.new_roi_button.clicked.connect(self.create_roi)
        roi_layout.addWidget(self.new_roi_button)
        roi_help = QLabel(
            "Paint this label with napari’s Labels brush. Use one static 3D "
            "mask across all phases; image and ROI grids must match."
        )
        roi_help.setWordWrap(True)
        roi_layout.addWidget(roi_help)
        layout.addWidget(roi_group)

        action_row = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze ROI")
        self.qc_button = QPushButton("Run study QC")
        self.analyze_button.clicked.connect(self.analyze)
        self.qc_button.clicked.connect(self.run_qc)
        action_row.addWidget(self.analyze_button)
        action_row.addWidget(self.qc_button)
        layout.addLayout(action_row)

        self.figure = Figure(figsize=(5.2, 4), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        self.signal_axes, self.enhancement_axes = self.figure.subplots(2, 1, sharex=True)
        self._empty_plot()
        layout.addWidget(self.canvas, stretch=1)
        plot_help = QLabel("Shading shows within-ROI voxel SD, not a confidence interval.")
        plot_help.setWordWrap(True)
        layout.addWidget(plot_help)

        self.qc_output = QPlainTextEdit()
        self.qc_output.setReadOnly(True)
        self.qc_output.setPlaceholderText("Quality-control results will appear here.")
        self.qc_output.setMaximumHeight(135)
        layout.addWidget(self.qc_output)
        self.export_button = QPushButton("Export CSV, plots, ROI and report…")
        self.export_button.clicked.connect(self.export_report)
        layout.addWidget(self.export_button)
        self.status_label = QLabel("Load the synthetic demo to try the complete workflow.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        self.image_combo.currentIndexChanged.connect(self._image_changed)
        self.roi_combo.currentIndexChanged.connect(self._invalidate)
        self.times_input.textChanged.connect(self._invalidate)
        self.baseline_spin.valueChanged.connect(self._invalidate)
        self.label_spin.valueChanged.connect(self._label_changed)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _selected_image(self) -> Image:
        layer = self.image_combo.currentData()
        if not isinstance(layer, Image) or layer not in self.viewer.layers:
            raise ValueError("Choose a 3D or 4D Image layer first.")
        return layer

    def _selected_roi(self) -> Labels:
        layer = self.roi_combo.currentData()
        if not isinstance(layer, Labels) or layer not in self.viewer.layers:
            raise ValueError("Choose or create a 3D Labels ROI first.")
        return layer

    def _disconnect_layer_events(self) -> None:
        for emitter, callback in self._connected_emitters:
            try:
                emitter.disconnect(callback)
            except (RuntimeError, ValueError):
                pass
        self._connected_emitters.clear()

    def _refresh_layers(self, *_args: object) -> None:
        old_image = self.image_combo.currentData()
        old_roi = self.roi_combo.currentData()
        self._disconnect_layer_events()
        self.image_combo.blockSignals(True)
        self.roi_combo.blockSignals(True)
        self.image_combo.clear()
        self.roi_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.ndim in (3, 4) and not layer.rgb:
                self.image_combo.addItem(layer.name, layer)
            if isinstance(layer, Labels) and layer.ndim == 3:
                self.roi_combo.addItem(layer.name, layer)
            for event_name in ("data", "paint", "scale", "translate", "rotate", "shear", "affine"):
                emitter = getattr(layer.events, event_name, None)
                if emitter is not None:
                    emitter.connect(self._invalidate)
                    self._connected_emitters.append((emitter, self._invalidate))
            layer.events.name.connect(self._refresh_layers)
            self._connected_emitters.append((layer.events.name, self._refresh_layers))
        if self.image_combo.findData(old_image) >= 0:
            self.image_combo.setCurrentIndex(self.image_combo.findData(old_image))
        if self.roi_combo.findData(old_roi) >= 0:
            self.roi_combo.setCurrentIndex(self.roi_combo.findData(old_roi))
        self.image_combo.blockSignals(False)
        self.roi_combo.blockSignals(False)
        if self.image_combo.currentData() is not old_image:
            self._image_changed()
        self._update_phase_count()
        self._invalidate()

    def _image_changed(self, *_args: object) -> None:
        self.times_input.clear()
        self._update_phase_count()
        self._invalidate()

    def _update_phase_count(self) -> None:
        image = self.image_combo.currentData()
        count = int(image.data.shape[0]) if isinstance(image, Image) and image.ndim == 4 else 1
        self.baseline_spin.setMaximum(max(count, 1))

    def _label_changed(self, *_args: object) -> None:
        roi = self.roi_combo.currentData()
        if isinstance(roi, Labels):
            roi.selected_label = self.label_spin.value()
        self._invalidate()

    def _invalidate(self, *_args: object) -> None:
        self.last_export = None
        self.last_curve = None
        if self._analysis_ready:
            self._analysis_ready = False
            self._empty_plot()
            self.qc_output.clear()
            self.status_label.setText(
                "Inputs changed. Analyze again; export always recomputes current data."
            )

    def _empty_plot(self) -> None:
        self.signal_axes.clear()
        self.enhancement_axes.clear()
        self.signal_axes.set_ylabel("Mean MRI signal (a.u.)")
        self.enhancement_axes.set_ylabel("Relative change (%)")
        self.enhancement_axes.set_xlabel("Phase")
        self.signal_axes.text(
            0.5,
            0.5,
            "Select an image and ROI, then analyze.",
            ha="center",
            va="center",
            transform=self.signal_axes.transAxes,
        )
        self.canvas.draw_idle()

    def _show_error(self, error: Exception) -> None:
        self._invalidate()
        self.status_label.setText(f"Error: {error}")

    def open_nifti(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open one 4D NIfTI or ordered 3D phases",
            "",
            "NIfTI images (*.nii *.nii.gz)",
        )
        if not paths:
            return
        if len(paths) > 1:
            order_dialog = _PhaseOrderDialog(paths, self)
            if order_dialog.exec() != QDialog.DialogCode.Accepted:
                return
            paths = order_dialog.paths
        try:
            study = load_nifti(paths)
            self._add_study(study)
            self.status_label.setText(
                f"Loaded {study.name}: {study.n_timepoints} phase(s). "
                "Create or select a Labels ROI, then paint your region."
            )
        except Exception as error:
            self._show_error(error)

    def _add_study(self, study: MRIStudy) -> Image:
        image = self.viewer.add_image(
            study.data,
            name=study.name,
            scale=(1.0, *study.spacing_zyx),
            metadata={"breast_mri_study": study},
            colormap="gray",
            rgb=False,
        )
        self.image_combo.setCurrentIndex(self.image_combo.findData(image))
        self.viewer.dims.ndisplay = 2
        axis_labels = list(self.viewer.dims.axis_labels)
        axis_labels[-4:] = ("phase", "z", "y", "x")
        self.viewer.dims.axis_labels = tuple(axis_labels)
        return image

    def load_demo(self) -> None:
        """Add a deterministic synthetic study and its static example ROI."""
        try:
            study = make_demo_study()
            image = self._add_study(study)
            roi = self.viewer.add_labels(
                make_demo_roi(study).astype(np.uint8),
                name="Demo ROI",
                scale=study.spacing_zyx,
                opacity=0.35,
            )
            self.image_combo.setCurrentIndex(self.image_combo.findData(image))
            self.roi_combo.setCurrentIndex(self.roi_combo.findData(roi))
            self.label_spin.setValue(1)
            self.analyze()
            if self._analysis_ready:
                self.status_label.setText(
                    "Synthetic demo ready. Scroll the phase axis, inspect the ROI "
                    "curve, or export a reproducible report."
                )
        except Exception as error:
            self._show_error(error)

    def create_roi(self) -> None:
        try:
            image = self._selected_image()
            study = self._study_from_layer(image)
            roi = self.viewer.add_labels(
                np.zeros(study.spatial_shape, dtype=np.uint32),
                name=f"{image.name} ROI",
                scale=tuple(image.scale[-3:]),
                translate=tuple(image.translate[-3:]),
                opacity=0.35,
            )
            # Reproduce the full spatial mapping, including rotation/shear.
            origin, basis = self._spatial_mapping(image)
            transform = np.eye(4)
            transform[:3, :3] = basis
            transform[:3, 3] = origin
            roi.scale = (1.0, 1.0, 1.0)
            roi.translate = (0.0, 0.0, 0.0)
            roi.affine = transform
            roi.selected_label = self.label_spin.value()
            self.roi_combo.setCurrentIndex(self.roi_combo.findData(roi))
            self.viewer.layers.selection.active = roi
            roi.mode = "paint"
            self.status_label.setText(
                f"Paint label {self.label_spin.value()} on the new ROI layer, "
                "then click Analyze ROI."
            )
        except Exception as error:
            self._show_error(error)

    def _study_from_layer(self, image: Image) -> MRIStudy:
        if image.multiscale or image.rgb:
            raise ValueError(
                "Multiscale and RGB images are not supported. Use one 3D/4D MRI array."
            )
        data = np.asarray(image.data)
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        if data.ndim != 4:
            raise ValueError("MRI images must use ZYX (3D) or TZYX (4D) axes.")
        original = image.metadata.get("breast_mri_study")
        if isinstance(original, MRIStudy) and tuple(original.data.shape) == tuple(data.shape):
            # Read live data, retaining verified source geometry and timing.
            study = replace(original, data=data, name=image.name)
        else:
            # napari's spatial axes are ZYX here. NIfTI indices/world axes
            # are represented as XYZ, so reverse both sides of the mapping.
            # This is a viewer-derived coordinate frame, not scanner anatomy.
            origin_zyx, basis_zyx = self._spatial_mapping(image)
            reverse_axes = np.eye(3)[::-1]
            viewer_affine_xyz = np.eye(4)
            viewer_affine_xyz[:3, :3] = reverse_axes @ basis_zyx @ reverse_axes
            viewer_affine_xyz[:3, 3] = reverse_axes @ origin_zyx
            study = MRIStudy(
                data=data,
                name=image.name,
                spacing_zyx=tuple(float(v) for v in np.linalg.norm(basis_zyx, axis=0)),
                affine_xyz=viewer_affine_xyz,
                warnings=(
                    "This existing layer is interpreted as ZYX/TZYX. "
                    "The ROI affine is reconstructed from viewer coordinates with "
                    "unknown spatial units; it is not a verified scanner anatomical frame. "
                    "No source NIfTI affine or verified acquisition times are available.",
                ),
            )
        timing_text = self.times_input.text().strip()
        if timing_text:
            try:
                times = np.asarray(
                    [float(value.strip()) for value in timing_text.split(",")],
                    dtype=np.float64,
                )
            except ValueError as error:
                raise ValueError("Enter acquisition seconds as comma-separated numbers.") from error
            study = replace(study, times_seconds=times)
        return study

    @staticmethod
    def _spatial_mapping(layer: Image | Labels) -> tuple[np.ndarray, np.ndarray]:
        origin_point = np.zeros(layer.ndim, dtype=float)
        world_origin = np.asarray(layer.data_to_world(tuple(origin_point)), dtype=float)
        origin = world_origin[-3:]
        columns = []
        for index in range(layer.ndim - 3, layer.ndim):
            point = origin_point.copy()
            point[index] = 1.0
            world_point = np.asarray(layer.data_to_world(tuple(point)), dtype=float)
            if layer.ndim == 4 and not np.isclose(
                world_point[0], world_origin[0], rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    "The image transform mixes space and time; reset it before ROI analysis."
                )
            columns.append(world_point[-3:] - origin)
        if layer.ndim == 4:
            later = origin_point.copy()
            later[0] = 1.0
            spatial_later = np.asarray(layer.data_to_world(tuple(later)), dtype=float)[-3:]
            if not np.allclose(spatial_later, origin, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "The image transform mixes time and space; reset it before ROI analysis."
                )
        basis = np.column_stack(columns)
        if (
            not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(basis))
            or np.linalg.matrix_rank(basis) < 3
        ):
            raise ValueError("The spatial transform must be finite and invertible.")
        return origin, basis

    def _compute_current(self) -> tuple[MRIStudy, np.ndarray, ROICurve, list[QCItem]]:
        image = self._selected_image()
        roi = self._selected_roi()
        study = self._study_from_layer(image)
        roi_data = np.asarray(roi.data)
        if roi_data.ndim != 3 or tuple(roi_data.shape) != tuple(study.spatial_shape):
            raise ValueError("ROI must be one 3D Labels array matching the image spatial shape.")
        image_origin, image_basis = self._spatial_mapping(image)
        roi_origin, roi_basis = self._spatial_mapping(roi)
        if not (
            np.allclose(image_origin, roi_origin, rtol=1e-6, atol=1e-5)
            and np.allclose(image_basis, roi_basis, rtol=1e-6, atol=1e-5)
        ):
            raise ValueError(
                "Image and ROI spatial transforms differ. Align them or create "
                "a new ROI from the selected image. No automatic registration is applied."
            )
        mask = roi_data == self.label_spin.value()
        if not np.any(mask):
            raise ValueError(
                f"Label {self.label_spin.value()} is empty. Paint an ROI or choose another label ID."
            )
        curve = analyze_roi(study, mask, baseline_index=self.baseline_spin.value() - 1)
        qc_items = qc_study(study)
        return study, mask, curve, qc_items

    def _render_qc(self, study: MRIStudy, items: list[QCItem]) -> None:
        lines = [f"{study.name} | {study.n_timepoints} phase(s) | ZYX {study.spatial_shape}"]
        lines.extend(f"[{item.severity.upper()}] {item.code}: {item.message}" for item in items)
        if not items:
            lines.append("No QC findings from the implemented checks.")
        self.qc_output.setPlainText("\n".join(lines))

    def _render_curve(self, curve: ROICurve) -> None:
        self.signal_axes.clear()
        self.enhancement_axes.clear()
        self.signal_axes.plot(curve.x, curve.mean, "o-", color="#4778d5", label="ROI mean")
        self.signal_axes.fill_between(
            curve.x,
            curve.mean - curve.std,
            curve.mean + curve.std,
            color="#4778d5",
            alpha=0.2,
            label="Within-ROI voxel SD",
        )
        self.signal_axes.set_ylabel("Mean MRI signal (a.u.)")
        self.signal_axes.legend(loc="best", fontsize="small")
        if np.isfinite(curve.relative_enhancement).any():
            self.enhancement_axes.plot(curve.x, curve.relative_enhancement, "o-", color="#cf6941")
            self.enhancement_axes.axhline(0.0, color="0.5", linewidth=0.8)
        else:
            self.enhancement_axes.text(
                0.5,
                0.5,
                "Relative change is undefined for this reference signal.",
                ha="center",
                va="center",
                wrap=True,
                transform=self.enhancement_axes.transAxes,
            )
        self.enhancement_axes.set_ylabel("Relative change (%)")
        self.enhancement_axes.set_xlabel(curve.x_label)
        self.signal_axes.grid(alpha=0.2)
        self.enhancement_axes.grid(alpha=0.2)
        self.canvas.draw_idle()

    def analyze(self) -> None:
        try:
            study, _mask, curve, items = self._compute_current()
            self._render_curve(curve)
            self._render_qc(study, items)
            self.last_curve = curve
            self._analysis_ready = True
            details = (
                f"{study.name} · label {self.label_spin.value()} · "
                f"{curve.voxel_count:,} ROI voxels · reference phase {self.baseline_spin.value()}."
            )
            if curve.warnings:
                details += " " + " ".join(curve.warnings)
            self.status_label.setText(details)
        except Exception as error:
            self._show_error(error)

    def run_qc(self) -> None:
        try:
            study = self._study_from_layer(self._selected_image())
            items = qc_study(study)
            self._render_qc(study, items)
            self.status_label.setText(
                "Study QC complete. Findings require review; these checks do not establish clinical suitability."
            )
        except Exception as error:
            self._show_error(error)

    def export_report(self) -> None:
        parent = QFileDialog.getExistingDirectory(self, "Choose a parent folder for the report")
        if not parent:
            return
        try:
            # The dialog may have been open while layers were edited. Always
            # compute again from the live image and mask after it closes.
            study, mask, curve, items = self._compute_current()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            destination = Path(parent) / f"breast-mri-report-{timestamp}"
            self.last_export = export_analysis(destination, study, mask, curve, items)
            self._render_curve(curve)
            self._render_qc(study, items)
            self.last_curve = curve
            self._analysis_ready = True
            self.status_label.setText(f"Exported {len(self.last_export)} files to {destination}")
        except Exception as error:
            self._show_error(error)

    def closeEvent(self, event: object) -> None:
        if self._viewer_events_connected:
            self.viewer.layers.events.inserted.disconnect(self._refresh_layers)
            self.viewer.layers.events.removed.disconnect(self._refresh_layers)
            self._viewer_events_connected = False
        self._disconnect_layer_events()
        super().closeEvent(event)
