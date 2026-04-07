# batcamp

[![Tests](https://github.com/svaberg/batcamp/actions/workflows/tests.yml/badge.svg)](https://github.com/svaberg/batcamp/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/batcamp.svg)](https://pypi.org/project/batcamp/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![DOI](https://zenodo.org/badge/1177665095.svg)](https://doi.org/10.5281/zenodo.19163499)

`batcamp` reconstructs octrees from BATSRUS-like outputs, with support for both spherical and cartesian data.

## What it does

- Rebuilds a usable octree from simulation output
- Provides fast, octree-aware resampling to planes, spheres, and rays

## Why

Some numerical codes provide leaf-cell values without storing the octree data. `batcamp` rebuilds that structure, permitting rapid interpolation and resampling.

## Installation

```bash
pip install batcamp
```

## Quick start

The following code plots a two-dimensional density slice.

```python
import numpy as np
import matplotlib.pyplot as plt

from batread import Dataset
from batcamp import Octree
from batcamp import OctreeInterpolator

# Read the dataset.
ds = Dataset.from_file("MY_FILE")
print(ds)

points = ds[["X [R]", "Y [R]", "Z [R]"]]
corners = ds.corners

# Build the octree and the interpolator.
octree = Octree(points, corners)
print(octree)
density_values = ds["Rho [g/cm^3]"]
interp = OctreeInterpolator(octree, density_values)
print(interp)

# Create a grid of points and interpolate the density.
X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
Z = np.zeros_like(X)
density = interp(X, Y, Z)
plt.pcolormesh(X, Y, density, norm="log")
```

For a fuller worked example, see
[examples/quick_start.ipynb](https://github.com/svaberg/batcamp/blob/main/examples/quick_start.ipynb)
in the repository.
