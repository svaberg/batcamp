<h1><img src="https://raw.githubusercontent.com/svaberg/batcamp/0.0.0/assets/batcamp.png" alt="batcamp logo"> batcamp</h1>

`batcamp` reconstructs octrees from BATSRUS-like outputs, with support for both spherical and cartesian data.

## What it does

- Rebuilds a usable octree from simulation output
- Supports saving and loading reconstructed octrees
- Fast, octree-aware resampling to e.g. planes, spheres, and rays

## Why

Many outputs provide leaf-cell values without storing the octree data. `batcamp` rebuilds that structure, permitting rapid interpolation and resampling.

## Quick start
After cloning the repository, the package may be installed with `pip` in the regular way. From the repository root run:

```bash
pip install .
```
The following code plots a two dimensional density slice.
```python
import numpy as np
import matplotlib.pyplot as plt

from batread import Dataset
from batcamp import OctreeBuilder
from batcamp import OctreeInterpolator

# Read the dataset
ds = Dataset.from_file("MY_FILE")

# Extract the mesh geometry explicitly.
points = np.column_stack((
    np.asarray(ds["X [R]"], dtype=float),
    np.asarray(ds["Y [R]"], dtype=float),
    np.asarray(ds["Z [R]"], dtype=float),
))
corners = np.asarray(ds.corners, dtype=np.int64)

# Build the octree from points and corners, then create the interpolator on top of it.
tree = OctreeBuilder().build(points, corners)
rho_values = np.asarray(ds["Rho [g/cm^3]"], dtype=float)
interp = OctreeInterpolator(tree, rho_values)

# Create a grid of points and interpolate the density
X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
Z = np.zeros_like(X)
rho = interp(X, Y, Z)
plt.pcolormesh(X, Y, rho, norm="log")
```

See the examples folder [examples/quick_start.ipynb](examples/quick_start.ipynb) for a running example.
