# Atomic-Scale Strain Mapping for Au/Cu Alloy Slabs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Robust ASE-based workflows for projected atomic-scale strain analysis in alloyed Au/Cu slab systems (as an example) with adsorbate-aware visualization, periodic replication, and publication-grade plotting.

This repository contains two complementary strain-analysis pipelines:

1. Voronoi surface strain mapping
2. Surface trimer strain mapping

Both implementations use a common fitted-surface-plane formalism to ensure that geometric measurements, strain calculations, and visualization are internally consistent.

Files:

* `atomic_scale_strain_mapping_voronoi_with_OH_O_markers.py`
* `atomic_scale_mapping_trimer_with_adsorbate_markers.py`

---

# Scientific Purpose

These scripts quantify net local geometric strain in alloyed Au/Cu surfaces (example) obtained from atomistic simulations stored in ASE databases (`.db`).

The workflows are designed for:

* alloy surface analysis,
* electrocatalytic interface studies,
* adsorbate-induced reconstruction analysis,
* strain/composition correlation studies,
* atomically resolved visualization of heterogeneous surfaces.



# Repository Structure

```text
.
├── atomic_scale_strain_mapping_voronoi_with_OH_O_markers.py
├── atomic_scale_mapping_trimer_with_adsorbate_markers.py
├── example.db
├── strain_ads_voronoi/
│   ├── strain_row_*.png
│   └── strain_ads_summary.csv
└── strain_ads_trimer/
    ├── trimer_strain_row_*.png
    └── strain_summary_trimers.csv
```

---

# Features

## Shared Core Features

Both workflows provide:

* ASE database integration
* Periodic boundary handling using minimum-image convention
* Surface-plane fitting using SVD
* Robust projected geometry analysis
* Au/Cu alloy-aware reference distances
* Automatic periodic tiling for visualization
* Adsorbate detection and annotation
* CSV summary export
* Diagnostic reporting
* Fault-tolerant row processing
* Symmetric logarithmic strain scaling

---

# Method 1 — Voronoi Area Strain Mapping

File:
`atomic_scale_strain_mapping_voronoi_with_OH_O_markers.py`

This workflow computes atom-centered local strain using Voronoi tessellation in the fitted surface plane.

## Physical Interpretation

Each surface atom receives a Voronoi polygon representing its local atomic territory.

Local strain is defined from:

* measured Voronoi area,
* reference elemental close-packed area.

The method is sensitive to:

* local compression/expansion,
* segregation,
* reconstruction,
* adsorbate-induced rearrangement.


## Strain Definition

Strain is reported as the percentage deviation of a measured local geometric quantity from its reference value:

$$
\varepsilon(\%) = 100 \times \frac{X - X_{\mathrm{ref}}}{X_{\mathrm{ref}}}
$$

where:

- \(X\) is the measured projected quantity in the fitted surface-plane basis.
- \(X_{\mathrm{ref}}\) is the corresponding reference value.
- Positive strain means net local expansion.
- Negative strain means net local compression.

For bond strain:

$$
\varepsilon_{\mathrm{bond}}(\%) =
100 \times \frac{d - d_{\mathrm{ref}}}{d_{\mathrm{ref}}}
$$

For Voronoi area strain:

$$
\varepsilon_{\mathrm{Voronoi}}(\%) =
100 \times \frac{A_{\mathrm{Voronoi}} - A_{\mathrm{ref}}}{A_{\mathrm{ref}}}
$$


## Outputs

### Images

* atomically resolved Voronoi strain maps,
* projected bond overlays,
* adsorbate annotations.

### CSV Summary

Includes:

* Au surface composition,
* subsurface composition,
* bond strain averages,
* Voronoi strain averages,
* plane-fit diagnostics.

---

# Method 2 — Trimer Area Strain Mapping

File:
`atomic_scale_mapping_trimer_with_adsorbate_markers.py`

This workflow computes strain from nearest-neighbor surface trimers.

## Physical Interpretation

Instead of atom-centered Voronoi regions, this method analyzes:

* triangular motifs,
* local three-atom surface units,
* projected trimer areas.

The approach is useful for:

* local coordination analysis,
* reconstruction detection,
* heterogeneous alloy motif characterization,
* surface topology studies.

## Trimer Construction

A trimer is accepted only if:

* all three pairwise bonds exist,
* all edges belong to the nearest-neighbor graph.

## Strain Definition


For trimer area strain:

$$
\varepsilon_{\mathrm{trimer}}(\%) =
100 \times \frac{A_{\mathrm{trimer}} - A_{\mathrm{trimer,ref}}}{A_{\mathrm{trimer,ref}}}
$$
Reference areas are computed using Heron's formula from elemental/mixed bond references.

## Supported Trimer Classes

* Au3
* Cu3
* Au2Cu
* AuCu2

## Outputs

### Images

* projected trimer strain maps,
* bond overlays,
* adsorbate markers.

### CSV Summary

Includes:

* bond strain statistics,
* trimer strain statistics by motif type,
* surface composition,
* plane diagnostics.

---

# Adsorbate Detection

Both scripts support automatic adsorbate annotation.

Assumptions:

* adsorbates use ASE tag `0`.

Detected species:

* OH oxygen → `x`
* isolated O → `^`

OH assignment uses:

* minimum-image O–H distance,
* threshold:
  `distance < 1.2 Å`

This allows rapid visual correlation between:

* adsorbates,
* local reconstruction,
* strain localization.

---

# Geometry and Projection Formalism

A key design feature is strict geometric consistency.

The workflow:

1. fits a local surface plane,
2. constructs orthonormal in-plane basis vectors,
3. projects all geometry into this basis.

This basis is used consistently for:

* bonds,
* Voronoi polygons,
* trimer polygons,
* plotting,
* distance calculations,
* area calculations.

Surface normal extraction uses SVD plane fitting.

---

# Reference Bond Lengths

Default references:

| Pair  | Distance (Å) |
| ----- | -----------: |
| Cu–Cu |        2.609 |
| Au–Au |        2.980 |
| Au–Cu |       2.7945 |

Defined in:

```python
D_REF = {
    ("Cu", "Cu"): 2.609,
    ("Au", "Au"): 2.980,
    ("Au", "Cu"): 2.7945,
}
```

Users should validate these references against:

* lattice constants,
* DFT setup,
* XC functional,
* relaxed bulk calculations.

---

# Requirements

## Python

Recommended:

* Python ≥ 3.10

## Dependencies

Install with:

```bash
pip install numpy matplotlib scipy shapely ase cmcrameri
```

Core packages:

* numpy
* matplotlib
* scipy
* shapely
* ase
* cmcrameri

---

# Input Requirements

The scripts expect:

* ASE `.db` database,
* periodic slab structures,
* tagged surface atoms.

Required conventions:

| Tag | Meaning         |
| --- | --------------- |
| 1   | Surface atom    |
| >1  | Subsurface/bulk |
| 0   | Adsorbates      |

Example:

```python
DB_PATH = "example.db"
```

---

# Usage

## Voronoi Workflow

```bash
python atomic_scale_strain_mapping_voronoi_with_OH_O_markers.py
```

## Trimer Workflow

```bash
python atomic_scale_mapping_trimer_with_adsorbate_markers.py
```

---

# Row Selection

Supported modes:

```python
ROW_IDS = "all"
```

```python
ROW_IDS = [3515, 3516]
```

```python
ROW_IDS = range(0, 200)
```

---

# Example Output

Generated outputs:

```text
strain_ads_voronoi/
├── strain_row_10.png
├── strain_row_11.png
└── strain_ads_summary.csv
```

```text
strain_ads_trimer/
├── trimer_strain_row_10.png
├── trimer_strain_row_11.png
└── strain_summary_trimers.csv
```

---

# Diagnostics

The scripts report:

* plane RMS deviation,
* maximum out-of-plane deviation,
* number of bonds,
* number of trimers,
* plotted polygons,
* skipped rows.

Example:

```text
plane RMS=0.0312 Å
max |deviation|=0.0814 Å
bonds=144
trimers=96
```

These diagnostics are important because strong out-of-plane distortion can invalidate purely 2D interpretations.

---

# Numerical and Geometric Considerations

## Important Assumptions

The workflows assume:

* approximately layered surfaces,
* identifiable surface plane,
* close-packed coordination.

Potential failure modes:

* highly corrugated slabs,
* amorphous surfaces,
* severe reconstruction,
* sparse surfaces,
* defective systems with broken local coordination.

---

# Robustness Features

Implemented protections include:

* invalid polygon handling,
* GeometryCollection support,
* periodic minimum-image distances,
* NaN-safe averaging,
* skip-on-failure row handling,
* finite-area validation,
* degeneracy filtering.

---

# Performance Notes

Performance scales primarily with:

* number of surface atoms,
* replication size,
* Voronoi complexity.

Most computational cost arises from:

* periodic tiling,
* Voronoi tessellation,
* polygon clipping.

For large datasets:

* reduce `MAX_REPEAT`,
* reduce `PLOT_L_VIEW`,
* process smaller row ranges.

---

# Recommended Validation

Before large-scale analysis:

1. Verify surface tagging.
2. Verify adsorbate tagging.
3. Validate reference distances.
4. Inspect several output plots manually.
5. Check plane RMS statistics.
6. Compare projected and raw geometries.

---

# Citation Guidance

If used in published work, cite:

* ASE,
* SciPy,
* Shapely,
* Matplotlib,
* cmcrameri.

Also describe:

* projected surface-plane methodology,
* reference bond calibration,
* local strain definition.

---

# License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

# Author Notes

The two workflows are complementary rather than interchangeable:

| Method  | Best For                    |
| ------- | --------------------------- |
| Voronoi | atom-centered local strain  |
| Trimer  | local motif/topology strain |

Using both together can help separate:

* local atomic packing effects,
* coordination topology changes,
* alloy motif distortions,
* adsorbate-induced restructuring.
