# v0.1.0 validation

Validated locally on **2026-10-24**, Windows, Python 3.12.14, napari 0.7.1 and
PySide6 6.12.0. See `requirements-tested.txt` for main dependency versions.

## Results

- **63 tests passed** with a real Windows Qt/OpenGL context.
- Numerical tests cover known ROI mean/SD, selected baselines, missing timing,
  nonfinite values, empty masks and invalid inputs.
- NIfTI tests cover canonical axes, temporal units, explicit file order,
  mismatched grids/units, complex-image rejection and narrow non-RGB volumes.
- Widget tests cover live image/ROI changes, painting invalidation, full spatial
  transforms, time/space coupling, the launcher and installed plugin discovery.
- Report tests cover CSV/JSON output, affine/unit preservation, stale input
  detection, zero reference signal and protection of existing exports.
- The real viewer rendered the synthetic image, Labels ROI and curves; its
  screenshot is in `docs/images/workbench.png`.
- The synthetic headless report command completed successfully.
- The npe2 manifest validated; wheel and source distributions built and passed
  `twine check`. The wheel was installed into a separate test target, and its
  packaged manifest and numerical analysis were exercised independently of `src/`.

The test run emitted upstream Pydantic `json_encoders` deprecation warnings.
Qt's offscreen backend could not create napari's OpenGL canvas on this Windows
machine; the native Windows backend was used instead. Linux CI uses Xvfb.

The first hosted CI runs also exposed two environment issues: Windows runners
need a software OpenGL implementation, and PySide6 6.12.0 on Python 3.11 passed
the assertions but crashed during interpreter shutdown. The GUI extra excludes
that PySide6 release on Python below 3.12; Python 3.12 was validated with 6.12.0.
Windows CI installs Mesa software OpenGL, following napari's own display setup.
These restrictions do not skip any tests.

## Evidence boundary

All bundled demonstration data are synthetic. These checks establish software
behavior on the tested cases, not clinical accuracy or robustness across MRI
scanners. No patient cohort, diagnostic endpoint, model performance or clinical
curve-classification claim was evaluated. The CI badge reports subsequent
Windows/Linux checks for each GitHub commit.
