# Dependencies and references

This project uses third-party packages through their public APIs. Their source
code and model weights are not vendored into this repository.

- [napari](https://github.com/napari/napari), BSD-3-Clause: image layers, viewer,
  Labels painting, and the plugin system.
- [napari plugin guide](https://napari.org/stable/plugins/building_a_plugin/index.html)
  and [plugin template](https://github.com/napari/napari-plugin-template): reference
  for the npe2 manifest and package structure. This repository's scaffold was
  written directly and was not copied from or generated with the template.
- [NiBabel](https://github.com/nipy/nibabel), MIT: NIfTI loading, orientation and writing.
- [NumPy](https://github.com/numpy/numpy), BSD-3-Clause: numerical arrays and statistics.
- [Matplotlib](https://github.com/matplotlib/matplotlib), Matplotlib license:
  signal plots and export figures.
- [QtPy](https://github.com/spyder-ide/qtpy), MIT, and
  [PySide6](https://doc.qt.io/qtforpython-6/licenses.html): widget toolkit and Qt binding.
  PySide6 has its own LGPL/GPL/commercial licensing terms.

The demo is synthetic. External clinical datasets, if used independently, retain
their own access conditions, licenses and citation requirements. For example,
[Duke-Breast-Cancer-MRI on TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)
is a possible source of research data, not a bundled component or a validation
dataset used for this release.
