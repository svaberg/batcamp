# Cross-Class Private Use Inventory

Scope: `batcamp/*.py`.

This file is about one thing only:

- which class or module-level function uses private state or private methods from some other class

So the provider names below are class/object names like `Octree`, not local variable names like `tree`.

Status labels:

- `acceptable`: natural construction-time or internal backend coupling
- `tolerable`: understandable, but it hard-codes another object's private layout
- `bad`: peer/runtime code reaching into another object's hidden state

## Cross-Class Use

| consumer | provider | location | status | why | private names |
| --- | --- | --- | --- | --- | --- |
| `CartesianOctreeBuilder.populate_tree_state` | `Octree` | `batcamp/builder_cartesian.py:390` | `acceptable` | builder is constructing the tree's exact internal state | `_corners`, `_points`, `_cell_centers`, `_cell_x_min`, `_cell_x_max`, `_cell_y_min`, `_cell_y_max`, `_cell_z_min`, `_cell_z_max`, `_xyz_min`, `_xyz_max`, `_i0`, `_i1`, `_i2`, `_node_depth`, `_node_i0`, `_node_i1`, `_node_i2`, `_node_value`, `_node_child`, `_root_node_ids`, `_node_parent`, `_cell_node_id`, `_node_x_min`, `_node_x_max`, `_node_y_min`, `_node_y_max`, `_node_z_min`, `_node_z_max`, `_radial_edges` |
| `SphericalOctreeBuilder.populate_tree_state` | `Octree` | `batcamp/builder_spherical.py:542` | `acceptable` | builder is constructing the tree's exact internal state | `_corners`, `_points`, `_cell_centers`, `_cell_r_min`, `_cell_r_max`, `_r_min`, `_r_max`, `_cell_polar_min`, `_cell_polar_max`, `_cell_azimuth_start`, `_cell_azimuth_width`, `_i0`, `_i1`, `_i2`, `_node_depth`, `_node_i0`, `_node_i1`, `_node_i2`, `_node_value`, `_node_child`, `_root_node_ids`, `_node_parent`, `_cell_node_id`, `_node_r_min`, `_node_r_max`, `_node_polar_min`, `_node_polar_max`, `_node_azimuth_start`, `_node_azimuth_width`, `_radial_edges` |
| `OctreeInterpolator.__init__` | `Octree` | `batcamp/interpolator.py:369` | `bad` | runtime peer object depends on hidden tree packaging | `_lookup_geometry` |
| `Octree.build_lookup` | `_CartesianCellLookup` or `_SphericalCellLookup` | `batcamp/octree.py:225` | `acceptable` | internal facade calling internal backend helpers in the same subsystem | `_init_lookup_state` |
| `Octree._cell_bounds_xyz` | `_CartesianCellLookup` or `_SphericalCellLookup` | `batcamp/octree.py:343` | `acceptable` | internal facade calling internal backend helpers in the same subsystem | `_cell_bounds_xyz` |
| `Octree._cell_bounds_rpa` | `_CartesianCellLookup` or `_SphericalCellLookup` | `batcamp/octree.py:347` | `acceptable` | internal facade calling internal backend helpers in the same subsystem | `_cell_bounds_rpa` |
| `Octree._domain_bounds_xyz` | `_CartesianCellLookup` or `_SphericalCellLookup` | `batcamp/octree.py:351` | `acceptable` | internal facade calling internal backend helpers in the same subsystem | `_domain_bounds_xyz` |
| `Octree._domain_bounds_rpa` | `_CartesianCellLookup` or `_SphericalCellLookup` | `batcamp/octree.py:355` | `acceptable` | internal facade calling internal backend helpers in the same subsystem | `_domain_bounds_rpa` |
| `OctreeArrayState.from_tree` | `Octree` | `batcamp/persistence.py:169` | `tolerable` | persistence naturally snapshots exact tree state, but it freezes private layout | `_i0`, `_i1`, `_i2`, `_node_depth`, `_node_i0`, `_node_i1`, `_node_i2`, `_node_value` |
| `OctreePersistenceState.instantiate_tree` | `Octree` | `batcamp/persistence.py:265` | `tolerable` | persistence rehydrates exact tree state, but it writes private layout directly | names from `_ARRAY_SPECS` and `_FLOAT_SCALAR_SPECS` |
| `_array_state_from_tree` | `Octree` | `batcamp/persistence.py:89` | `tolerable` | persistence reads exact tree state, but it is coupled to private layout | names from `_ARRAY_SPECS` and `_FLOAT_SCALAR_SPECS` |
| `build_face_neighbors` | `Octree` | `batcamp/face_neighbors.py:384` | `bad` | peer runtime code derives topology by calling a private tree method | `_frontier_nodes` |
| `_node_point_candidates` | `Octree` | `batcamp/ray.py:251` | `bad` | ray helper depends on hidden tree geometry packaging | `_lookup_geometry` |
| `_build_cell_plane_kernel_state` | `Octree` | `batcamp/ray.py:486` | `bad` | ray helper depends on hidden tree geometry packaging | `_lookup_geometry` |
| `_build_cartesian_ray_cell_geometry` | `Octree` | `batcamp/ray.py:584` | `bad` | ray helper depends on hidden tree geometry packaging | `_lookup_geometry` |
| `_build_spherical_ray_cell_geometry` | `Octree` | `batcamp/ray.py:675` | `bad` | ray helper depends on hidden tree geometry packaging | `_lookup_geometry` |
| `_build_sparse_spherical_seed_lookup_state` | `Octree` | `batcamp/ray.py:788` | `bad` | ray helper depends on hidden tree geometry packaging | `_lookup_geometry` |
| `OctreeRayTracer.__init__` | `Octree` | `batcamp/ray.py:2212` | `bad` | runtime peer object depends on hidden tree geometry packaging | `_lookup_geometry` |
| `OctreeRayInterpolator.__init__` | `OctreeInterpolator` | `batcamp/ray.py:2534` | `bad` | runtime peer object reaches into interpolator internals for backend state | `_interp_state_rpa` |
| `OctreeRayInterpolator.integrate_field_along_rays` | `OctreeRayTracer` | `batcamp/ray.py:2567` | `bad` | runtime peer object reaches directly into tracer internals instead of using an explicit surface | `_face_neighbor_state`, `_seed_lookup_state`, `_cell_lookup_state`, `_seed_cell_plane_state`, `_cell_plane_state` |

## Notes

- This is the cross-class list.
- It intentionally does not list ordinary `self._...` state access.
- It also does not list same-class helper calls like `_SphericalCellLookup._path(...)` unless the consumer and provider are different classes.
