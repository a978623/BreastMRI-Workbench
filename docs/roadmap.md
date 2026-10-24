# Roadmap / 可以继续魔改的方向

## v0.1: implemented research prototype

- NIfTI reader, deterministic synthetic sample and a napari widget.
- Static 3D Labels ROI, signal/enhancement curves and observable quality checks.
- Portable exports and tests for numerical/geometry/time-unit behavior.

## Next: usability and engineering

1. Background loading/analysis with progress, cancellation and memory limits.
2. Explicit irregular acquisition-time editor with provenance and validation.
3. Import pre-existing ROI NIfTI files with affine-aware grid checks.
4. Configurable multi-ROI batch exports and a study manifest.
5. Windows packaging and a clean-environment installation walkthrough.

## Later: research extensions

1. DICOM series discovery with physical slice order and explicit de-identification policy.
2. Motion review and optional externally validated registration workflows.
3. Separate linked T2/DWI viewers with spatial alignment checks.
4. Reproducible radiomics or prediction experiments with patient-level splits.

These are planned extensions, not claims about v0.1. When making a new feature,
add a small test dataset demonstrating the problem, implement the feature, and
record the before/after behavior in the README or release notes.
