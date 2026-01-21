# MAILDE_STRAIN_MAP

# Surface Strain Mapping via Voronoi Analysis and Plane-Projected Bond Strain

## Overview

This code computes and visualizes **surface strain** in metallic slab systems
(e.g. Cu–Au alloys) stored in an ASE database. Two complementary strain measures are
evaluated:

1. **Surface area strain** derived from Wigner–Seitz (Voronoi) polygons
2. **Bond strain** derived from nearest-neighbor bonds projected onto the best-fit
   surface plane

The result for each configuration is a **2D strain map** consisting of:
- Colored Voronoi polygons (area strain)
- Colored surface bridge bonds (bond strain)
- Atomic positions overlaid for reference

Each database row is processed independently, and output is written as an SVG figure.

---

## Key Features

- Robust plane fitting using a median-z filter and SVD
- Physically meaningful surface areas via plane-projected correction
- Bond strain computed from plane-projected bond vectors (not raw 3D distance)
- Lattice-vector–based 2D projection (no PCA, no rotational ambiguity)
- Reference normalization against pure-element slabs (Cu, Au)
- Symmetric-log color scaling for strain visualization
- Periodic replication with windowed visualization

---

## Input Requirements

### ASE Database

The code expects an ASE `.db` file containing slab structures with:

- Valid periodic boundary conditions
- Atomic **tags** identifying layers:
  - `SURFACE_TAG` (default: `1`) → surface layer
  - `SURFACE_TAG + 1` → subsurface layer
- At least:
  - One *pure Cu* slab
  - One *pure Au* slab  
  These are used as reference systems for strain normalization.

Example:
```python
DB_PATH = "test.db"
