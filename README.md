<h1><img src="https://raw.githubusercontent.com/svaberg/batcamp/main/assets/batcamp.png" alt="batcamp logo"> batcamp</h1>

[![Tests](https://github.com/svaberg/batcamp/actions/workflows/tests.yml/badge.svg)](https://github.com/svaberg/batcamp/actions/workflows/tests.yml) [![PyPI version](https://badge.fury.io/py/batcamp.svg)](https://pypi.org/project/batcamp/) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard) [![Codacy Badge](https://app.codacy.com/project/badge/Coverage/ac23b58aa5f14f9098d47c63ef054a63)](https://app.codacy.com/gh/svaberg/batcamp/dashboard) [![DOI](https://zenodo.org/badge/1177665095.svg)](https://doi.org/10.5281/zenodo.19163499)

`batcamp` reconstructs traversable octrees from BATSRUS-like outputs, and supports both spherical and cartesian data.

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
The octree interpolator interface is made to resemble the [`scipy` `LinearNDInterpolator`](https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.LinearNDInterpolator.html). The following code plots a two-dimensional density slice.
```python
# Read an example dataset using the batread library
from batread import Dataset
ds = Dataset.from_file("sample_data/3d__var_1_n00000000.plt")
print(ds)

# Build the octree from the dataset points and corners.
from batcamp import Octree
octree = Octree(
    points=ds[["X [R]", "Y [R]", "Z [R]"]],
    corners=ds.corners)
print(octree)

# Build the interpolator object from the octree and data values
from batcamp import OctreeInterpolator
octree_interpolator = OctreeInterpolator(octree, values=ds["Rho [g/cm^3]", "ti [K]"])
print(octree_interpolator)

# Create a regular grid of points, interpolate the data 
# onto them, and display the result with matplotlib
import numpy as np
import matplotlib.pyplot as plt
X, Y = np.meshgrid(np.linspace(-24, 24, 512), np.linspace(-24, 24, 512))
Z = np.zeros_like(X)
rho_and_ti = octree_interpolator(X, Y, Z)
plt.pcolormesh(X, Y, rho_and_ti[..., 0], norm="log")
plt.colorbar()
plt.show()
```

See [examples/quick_start.ipynb](examples/quick_start.ipynb) for a larger example.

## Synthetic images
For cartesian data, you can create a synthetic image from a set of rays.

```python
ds = Dataset.from_file("sample_data/3d__var_2_n00006003.plt")
tree = Octree.from_ds(ds, tree_coord="xyz")
interp = OctreeInterpolator(tree, ds["Rho [g/cm^3]"])
tracer = OctreeRayTracer(tree)

y, z = np.meshgrid(
    np.linspace(-24, 24, 512),
    np.linspace(-24, 24, 512),
    indexing="xy",
)
origins = np.stack((np.full_like(y, -24), y, z), axis=-1)
image, _ = tracer.accumulate_exact_image(interp, origins, np.array([1.0, 0.0, 0.0]))

plt.pcolormesh(y, z, image, shading="auto", norm="log")
plt.colorbar(label="line integral")
plt.show()
```
