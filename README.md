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
plt.title("Equatorial slice")
plt.colorbar()
plt.show()
```

See [examples/quick_start.ipynb](examples/quick_start.ipynb) for a larger example.

## Synthetic images
`batcamp` also provides a specialised ray tracer, which can be used e.g. to create synthetic images.

```python
ds = Dataset.from_file("sample_data/3d__var_2_n00060005.plt")
tree = Octree.from_ds(ds)
interp = OctreeInterpolator(tree, ds["Rho [g/cm^3]"])

from batcamp import OctreeRayTracer
octree_ray_tracer = OctreeRayTracer(tree)
print(octree_ray_tracer)

# Create rays that will form the image
x, z = np.meshgrid(
    np.linspace(-4, 4, 512),
    np.linspace(-4, 4, 512),
    indexing="xy")
y = -24 * np.ones_like(x)
origins_xyz = np.stack((x, y, z), axis=-1)
direction = np.array([0.0, 1.0, 0.0])

# Create the image and show it
image, _ = octree_ray_tracer.trilinear_image(interp, origins_xyz, direction)
plt.pcolormesh(x, z, image, norm="log")
plt.colorbar()
plt.title("Synthetic image")
plt.show()
```

A raytracing example using a larger, more realistic data set can be seen in [examples/ray_image.ipynb](examples/ray_image.ipynb).
