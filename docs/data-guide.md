# Data guide / 数据使用说明

## Image axes and physical geometry

NIfTI files are read with NiBabel and reoriented with `as_closest_canonical`.
NIfTI's `(X, Y, Z, T)` axes become `(T, Z, Y, X)` in napari. A 3D input gains a
single phase axis. Canonicalization can permute/flip axes; it does **not** perform
motion correction or register different acquisitions.

All separately loaded phases must have compatible canonical spatial shapes and
affines. If that check fails, the loader reports an error. Align or resample data
in a separate validated preprocessing workflow before using it here.

The napari display uses voxel spacing, with the canonical affine retained in the
study metadata for ROI export. The display is not a complete patient-coordinate
navigation system, especially for oblique acquisitions. Labels must use the same
spatial scale/translation as the image. The widget rejects mismatched transforms.

## Timing and baseline

For one 4D NIfTI, finite positive temporal spacing plus a known time unit
(`sec`, `msec`, `usec`) can supply relative timestamps. These are offsets from
the first phase, not injection times. Unknown units and separately selected
volumes use 1-based phase indices. If acquisition is irregular, use the Python
API's explicit `times_seconds` input after verifying metadata.

Files selected together retain their supplied order. No file-name sorting or
automatic identification of pre-contrast/early/late phases is performed.
Choose the actual pre-contrast phase as the baseline in the widget. A selected
baseline with a near-zero mean cannot support a stable relative signal ratio.

DCE phases are time points, not separate imaging modalities. v0.1 analyzes one
co-registered signal series. Combining T2/DWI/clinical information is a future
extension with its own preprocessing and validation requirements.

## ROI statistics

A static 3D label mask is applied to every phase. Mean and population standard
deviation are calculated over finite voxels in that mask for each phase. A phase
with no finite ROI values is an error. Invalid image values remain visible to QC.

The mean signal depends on the acquired sequence, scaling, motion and selected
ROI. Neither enhancement nor the curve shape is used to diagnose disease.

## Export and reproducibility

Export the binary mask alongside the CSV/JSON results. The mask is written as
canonical `(X,Y,Z)` NIfTI with the study affine, so the selected region can be
inspected independently. The report omits input paths and filenames. Exporting
again requires a new folder; outputs from earlier analyses are preserved.

Use the synthetic demo for development and public examples. Real images remain
outside the repository; `data/`, `exports/`, DICOM and NIfTI files are Git-ignored.
