<h1><img src="https://raw.githubusercontent.com/svaberg/batcamp/0.0.0/assets/batcamp.png" alt="batcamp logo"> batcamp</h1>

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

from batread.dataset import Dataset
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
