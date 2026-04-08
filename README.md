<h1><img src="https://raw.githubusercontent.com/svaberg/batcamp/0.0.0/assets/batcamp.png" alt="batcamp logo"> batcamp</h1>

[![Tests](https://github.com/svaberg/batcamp/actions/workflows/tests.yml/badge.svg)](https://github.com/svaberg/batcamp/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/batcamp.svg)](https://pypi.org/project/batcamp/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![DOI](https://zenodo.org/badge/1177665095.svg)](https://doi.org/10.5281/zenodo.19163499)

`batcamp` reconstructs traversable octrees from BATSRUS-like outputs, and supports for both spherical and cartesian data.

## What it does

- Rebuilds a usable octree from simulation output, and
- provides fast, octree-aware resampling to e.g. planes, spheres, and rays.

## Why

Some numerical codes provide leaf-cell values without storing the octree data. `batcamp` rebuilds that structure, permitting rapid interpolation and resampling.

## Installation
After cloning the repository, the package may be installed with `pip` in the regular way:

```bash
cd batcamp
pip install --editable .
```
The `--editable` flag is only required if you want to edit the code.

## Quick start
The octree interpolator interface is made to resemble the [`scipy` `LinearNDInterpolator`](https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.LinearNDInterpolator.html). The following code plots a two dimensional density slice.
```python
#
# Read the dataset using the batread library
#
from batread import Dataset
ds = Dataset.from_file("MY_FILE")
print(ds)
points = ds[["X [R]", "Y [R]", "Z [R]"]]  # Point-coordinate array with shape (n_points, 3).
corners = ds.corners  # Cell-to-corner connectivity with shape (n_cells, 8).

#
# Build the octree from points and corners
# 
from batcamp import Octree
octree = Octree(points, corners)
print(octree)

#
# Build the interpolator object from the octree
#
from batcamp import OctreeInterpolator
density_values = ds["Rho [g/cm^3]"]  # Shape (n_points, n_var), here n_var=1
octree_interpolator = OctreeInterpolator(octree, density_values)
print(octree_interpolator)

#
# Create a regular grid of points, interpolate the density 
# onto them, and display the result with matplotlib
#
import numpy as np
import matplotlib.pyplot as plt
X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
Z = np.zeros_like(X)
density = octree_interpolator(X, Y, Z)
plt.pcolormesh(X, Y, density, norm="log")
plt.show()
```

See the examples folder [examples/quick_start.ipynb](examples/quick_start.ipynb) for a running example.
