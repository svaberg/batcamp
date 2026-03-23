<h1><img src="https://raw.githubusercontent.com/svaberg/batcamp/0.0.0/assets/batcamp.png" alt="batcamp logo"> batcamp</h1>

[![Tests](https://github.com/svaberg/batcamp/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/svaberg/batcamp/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/batcamp.svg)](https://pypi.org/project/batcamp/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![DOI](https://zenodo.org/badge/1177665095.svg)](https://doi.org/10.5281/zenodo.19163499)

`batcamp` reconstructs octrees from BATSRUS-like outputs, with support for both spherical and cartesian data.

## What it does

- Rebuilds a usable octree from simulation output
- Supports saving and loading reconstructed octrees
- Fast, octree-aware resampling to e.g. planes and spheres

## Why

Many outputs provide leaf-cell values without storing the octree data. `batcamp` rebuilds that structure, permitting rapid interpolation and resampling.

## Quick start
Install with `pip`:

```bash
pip install batcamp
```
The examples below use `batread` to read BATSRUS/Tecplot files.

The following code plots a two dimensional density slice.
```python
import numpy as np
import matplotlib.pyplot as plt

from batread import Dataset
from batcamp import OctreeInterpolator

# Read the dataset
ds = Dataset.from_file("MY_FILE")

# Create the interpolator; the octree is built automatically.
interp = OctreeInterpolator(ds, ["Rho [g/cm^3]"])

# Create a grid of points and interpolate the density
X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
Z = np.zeros_like(X)
rho = interp(X, Y, Z)
plt.pcolormesh(X, Y, rho, norm="log")
```

See the running example notebook: [examples/quick_start.ipynb](https://github.com/svaberg/batcamp/blob/main/examples/quick_start.ipynb).
