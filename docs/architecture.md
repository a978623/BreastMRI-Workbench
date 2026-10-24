# Architecture

```text
NIfTI → io.load_nifti → MRIStudy (TZYX, spacing, canonical affine, timing)
                             ↓
                    napari image + Labels ROI
                             ↓
              analyze_roi + qc_study (pure numerical core)
                             ↓
          widget plots / CSV + JSON + PNG + ROI NIfTI + HTML
```

- `core.py`: validated study model, ROI statistics, explicit QC observations.
- `io.py`: NIfTI orientation/time-unit handling and the npe2 reader contribution.
- `sample_data.py`: deterministic synthetic study and reproducible ROI.
- `_widget.py`: layer selection, ROI creation, interaction and actionable errors.
- `report.py`: portable outputs; no web API calls and no implicit patient identifiers.
- `napari.yaml`: npe2 registration of reader, widget and synthetic sample.
- `__main__.py`: GUI launcher and a headless synthetic-report command.

Scientific calculations are separate from Qt so they can be tested on small,
known arrays. GUI tests exercise real napari layers, invalid selections and
export behavior. The reader does not guess acquisition chronology or resample
incompatible grids. Export should always use current layer and ROI data.

This first version keeps the pipeline local and synchronous. Very large studies
can pause the GUI during loading and calculations. A future worker-thread design
should add cancellation and bounded-memory operations before supporting large
batch workflows.
