# fidA

**fidA** is a Python package for processing Free Induction Decay (FID) data, inspired by the FID-A MATLAB code by Jamie Near.

## Authors

- **Colleen Bailey**, Sunnybrook Research Institute
- **Ira Yao**

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Importable Classes and Functions](#importable-classes-and-functions)
- [Publishing as a Python Package](#publishing-as-a-python-package)
- [Running Tests](#running-tests)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Data Import/Export**:
  - Load FID data from various formats (e.g., Bruker).
  - Read and write LCModel coordinate files.
  - Write data in LCModel `.RAW` format.

- **Processing Functions**:
  - Phase correction (zero and first-order).
  - Frequency alignment and shifting.
  - Averaging and median filtering of scans.
  - Exponential line broadening filter.
  - Zero-padding for spectral resolution enhancement.
  - Automatic referencing of ppm values.

- **Peak Fitting**:
  - Fit single or multiple peaks using Lorentzian or Gaussian models.
  - Estimate peak parameters like amplitude, full width at half maximum (FWHM), and chemical shift.

- **Visualization**:
  - Plot spectra with customizable parameters.
  - Ridge plots for comparing multiple spectra.

## Installation

You can install `fidA` from PyPI (once it's published) or from the source code.

### From PyPI (Not yet available)

```bash
pip install fidA
