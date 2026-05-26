#MAILDE S OZORIO - ATOMIC SCALE STRAIN MAPPING

"""
Atomic-scale strain mapping for Au/Cu slab ASE databases.

Upgrades relative to the original script
----------------------------------------
1. ASE-native row selection: uses db.select() rows directly.
2. Voronoi and bond strain are computed in the same fitted surface plane.
3. CSV summary is written and flushed after each processed row.
4. Robust Shapely clipping: supports Polygon, MultiPolygon, and GeometryCollection.
5. Explicit diagnostics and skip reporting.
6. No db.get(id=...) reconstruction and no generator/yield row-id bug.

Assumptions
-----------
- Surface atoms are tagged with SURFACE_TAG.
- Surface atoms form an approximately close-packed 2D layer.
- Bond strain is computed from projected nearest-neighbor distances in the fitted
  surface plane.
- Voronoi area strain is computed from 2D Voronoi cells in that same fitted plane.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic
from scipy.spatial import Voronoi, QhullError

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator

from shapely.geometry import Polygon as SPoly, box
from shapely.geometry.base import BaseGeometry

try:
    from cmcrameri import cm
    CMAP = cm.vik
except Exception:
    # Fallback so the script remains runnable if cmcrameri is not installed.
    CMAP = plt.get_cmap("coolwarm")


# ============================================================
# USER SETTINGS
# ============================================================

DB_PATH = "example.db"

# Row selection examples:
# ROW_IDS = "all"
# ROW_IDS = [3515, 3516, 3520]
#ROW_IDS = range(4309, 5600)
ROW_IDS =  range(0, 200) #Plot the a range of row id, to plot all set "all" or choose the id (examples above)

SURFACE_TAG = 1

N_NEIGHBORS = 6
LINEWIDTH = 5.0
ATOM_SIZE = 127.0

PLOT_N_ATOMS = 7
PLOT_SCALE = 1.10
PLOT_L_OVERRIDE = None

PLOT_L_VIEW = 15.0
MAX_REPEAT = 12

ATOM_COLORS = {"Au": "orange", "Cu": "brown"}
EPS = 1e-8

OUTDIR = Path("strain_ads_voronoi")
OUTDIR.mkdir(exist_ok=True)
CSV_PATH = OUTDIR / "strain_ads_summary.csv"

D_REF = {
    ("Cu", "Cu"): 2.609,
    ("Au", "Au"): 2.980,
    ("Au", "Cu"): 2.7945,
}

CB_VMAX = 20.0
CB_LINTHRESH = 5.0
CB_LINSCALE = 2.5

BOND_WIDTH_MAIN = LINEWIDTH
BOND_WIDTH_OUTLINE = LINEWIDTH + 1.1
BOND_OUTLINE_COLOR = "black"

Z_TOL_PLANE = 0.57


# ============================================================
# BASIC HELPERS
# ============================================================

def require_file(path: str | Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Database file does not exist: {path}")


def ref_bond_length(a: str, b: str) -> float:
    return D_REF.get(tuple(sorted((a, b))), np.nan)


def mic_vec(atoms, i: int, j: int) -> np.ndarray:
    vec, _ = find_mic(
        atoms.positions[j] - atoms.positions[i],
        atoms.get_cell(),
        atoms.get_pbc(),
    )
    return np.asarray(vec, dtype=float)


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if len(values) else float("nan")


def au_composition(atoms, surface_tag: int = 1) -> tuple[float, float]:
    tags = np.asarray(atoms.get_tags())
    syms = np.asarray(atoms.get_chemical_symbols())

    surf = syms[tags == surface_tag]
    subsurf = syms[tags > surface_tag]

    def frac_au(arr: np.ndarray) -> float:
        return 100.0 * np.sum(arr == "Au") / len(arr) if len(arr) else 0.0

    return frac_au(surf), frac_au(subsurf)


# ============================================================
# ROW SELECTION
# ============================================================

def select_rows(db, row_ids):
    """
    Select ASE database rows using db.select() row objects directly.
    """
    all_rows = list(db.select())

    if len(all_rows) == 0:
        raise RuntimeError("Database contains zero rows.")

    print("=" * 56)
    print("Database diagnostics")
    print("=" * 56)
    print(f"Rows available : {len(all_rows)}")
    print(f"Min row id     : {min(r.id for r in all_rows)}")
    print(f"Max row id     : {max(r.id for r in all_rows)}")
    print(f"First 20 ids   : {[r.id for r in all_rows[:20]]}")

    if row_ids == "all":
        rows = all_rows
    elif isinstance(row_ids, list):
        if not all(isinstance(i, int) for i in row_ids):
            raise TypeError("All ROW_IDS list elements must be integers.")
        valid_ids = set(row_ids)
        rows = [row for row in all_rows if row.id in valid_ids]
    elif isinstance(row_ids, range):
        valid_ids = set(row_ids)
        rows = [row for row in all_rows if row.id in valid_ids]
    else:
        raise ValueError("ROW_IDS must be 'all', list[int], or range.")

    if len(rows) == 0:
        raise RuntimeError("Row selection returned zero rows. Check ROW_IDS.")

    print(f"Rows selected  : {len(rows)}")
    print("=" * 56)
    print()
    return rows


# ============================================================
# PLANE GEOMETRY
# ============================================================

def median_filtered_plane_normal(pos3d: np.ndarray, z_tol: float = Z_TOL_PLANE) -> np.ndarray:
    pos3d = np.asarray(pos3d, dtype=float)

    if len(pos3d) < 3:
        return np.array([0.0, 0.0, 1.0])

    z = pos3d[:, 2]
    z0 = np.median(z)
    mask = np.abs(z - z0) < z_tol
    pts = pos3d[mask]

    if len(pts) < 3:
        pts = pos3d

    origin = pts.mean(axis=0)
    x = pts - origin
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    n = vt[-1]
    nn = np.linalg.norm(n)

    if nn < 1e-12:
        return np.array([0.0, 0.0, 1.0])

    n = n / nn

    # Keep orientation consistent with +z when possible.
    if n[2] < 0:
        n = -n

    return n


def plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal 2D basis vectors spanning the plane normal to n."""
    n = np.asarray(n, dtype=float)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError("Cannot construct plane basis from near-zero normal.")
    n = n / nn

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = ref - np.dot(ref, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    return e1, e2


def project_positions_to_plane(pos3d: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    pos3d = np.asarray(pos3d, dtype=float)
    r = pos3d - np.asarray(origin, dtype=float)
    return np.column_stack((r @ e1, r @ e2))


def project_vector_to_plane_2d(vec3d: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    vec3d = np.asarray(vec3d, dtype=float)
    return np.array([vec3d @ e1, vec3d @ e2], dtype=float)


def plane_deviation_stats(pos3d: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """
    Return RMS and maximum absolute distance of atoms from the fitted plane.

    Distances are signed projections along the plane normal:
        d_i = (r_i - origin) · normal

    Returns
    -------
    plane_rms_A : float
        Root-mean-square distance from the fitted plane, in Å.

    plane_max_abs_A : float
        Maximum absolute distance from the fitted plane, in Å.
    """
    pos3d = np.asarray(pos3d, dtype=float)
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)

    nn = np.linalg.norm(normal)
    if nn < 1e-12:
        raise ValueError("Cannot compute plane deviation from near-zero normal.")

    normal = normal / nn
    d = (pos3d - origin) @ normal

    plane_rms_A = float(np.sqrt(np.mean(d**2)))
    plane_max_abs_A = float(np.max(np.abs(d)))

    return plane_rms_A, plane_max_abs_A


# ============================================================
# SURFACE REPLICATION AND GEOMETRY
# ============================================================

def repeat_surface(surface_atoms, nx: int, ny: int):
    pos = np.asarray(surface_atoms.positions, dtype=float)
    cell = surface_atoms.get_cell()
    syms = surface_atoms.get_chemical_symbols()

    all_pos = []
    all_sym = []
    tile_id = []

    for ix in range(nx):
        for iy in range(ny):
            shift = ix * cell[0] + iy * cell[1]
            all_pos.append(pos + shift)
            all_sym.extend(syms)
            tile_id.extend([(ix, iy)] * len(pos))

    return np.vstack(all_pos), all_sym, np.asarray(tile_id, dtype=int)


def find_central_surface_atom(surface_atoms, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> int:
    p2d = project_positions_to_plane(surface_atoms.positions, origin, e1, e2)
    c = p2d.mean(axis=0)
    return int(np.argmin(np.sum((p2d - c) ** 2, axis=1)))


def order_polygon_ccw(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    c = poly.mean(axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iter_polygons(geom: BaseGeometry):
    """Yield Polygon geometries from Polygon/MultiPolygon/GeometryCollection."""
    if geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for g in geom.geoms:
            yield from iter_polygons(g)


def estimate_nn_distance(surface_atoms, center_index: int, e1: np.ndarray, e2: np.ndarray) -> float:
    distances = []
    for j in range(len(surface_atoms)):
        if j == center_index:
            continue
        vec = mic_vec(surface_atoms, center_index, j)
        v2d = project_vector_to_plane_2d(vec, e1, e2)
        d = np.linalg.norm(v2d)
        if d > EPS:
            distances.append(d)

    if len(distances) == 0:
        raise RuntimeError("Could not estimate nearest-neighbor distance.")

    distances = np.sort(distances)
    return float(np.median(distances[: min(N_NEIGHBORS, len(distances))]))


# ============================================================
# REFERENCE VORONOI AREAS
# ============================================================

def ref_voronoi_area_from_bond_length(symbol: str) -> float:
    """
    Elemental 2D Voronoi area for a close-packed triangular surface lattice.

    A = sqrt(3) / 2 * d^2
    """
    pair = (symbol, symbol)
    if pair not in D_REF:
        raise ValueError(f"No elemental reference bond length available for {symbol!r}")

    d = D_REF[pair]
    return 0.5 * np.sqrt(3.0) * d**2


# ============================================================
# BONDS
# ============================================================

def build_bridge_bonds(surface_atoms, e1: np.ndarray, e2: np.ndarray):
    syms = surface_atoms.get_chemical_symbols()
    n = len(surface_atoms)
    bond_dict = {}

    for i in range(n):
        cand = []

        for j in range(n):
            if i == j:
                continue

            vec = mic_vec(surface_atoms, i, j)
            v2d = project_vector_to_plane_2d(vec, e1, e2)
            d = np.linalg.norm(v2d)
            cand.append((d, j, vec))

        cand.sort(key=lambda x: x[0])

        for d, j, vec in cand[:N_NEIGHBORS]:
            a, b = sorted((i, j))

            if (a, b) in bond_dict:
                continue

            d0 = ref_bond_length(syms[i], syms[j])
            if not np.isfinite(d0) or d0 <= 0:
                continue

            pct = 100.0 * (d - d0) / d0
            bond_dict[(a, b)] = {
                "i": a,
                "j": b,
                "vec_ab": vec if a == i else -vec,
                "pct_dev": pct,
            }

    return list(bond_dict.values())


# ============================================================
# ROW ANALYSIS
# ============================================================

def analyze_and_plot_row(row, area_ref_by_symbol: dict[str, float], norm: SymLogNorm):
    atoms = row.toatoms()
    au_surf, au_sub = au_composition(atoms, SURFACE_TAG)

    tags = np.asarray(atoms.get_tags())
    surface_atoms = atoms[tags == SURFACE_TAG]

    if len(surface_atoms) < 3:
        raise RuntimeError(f"Need at least 3 surface atoms; found {len(surface_atoms)}.")

    surface_syms = surface_atoms.get_chemical_symbols()
    n_surf = len(surface_atoms)

    n_plane = median_filtered_plane_normal(surface_atoms.positions, Z_TOL_PLANE)
    origin_plane = np.asarray(surface_atoms.positions, dtype=float).mean(axis=0)
    e1, e2 = plane_basis_from_normal(n_plane)

    plane_rms_A, plane_max_abs_A = plane_deviation_stats(
        surface_atoms.positions,
        origin_plane,
        n_plane,
    )

    i_center = find_central_surface_atom(surface_atoms, origin_plane, e1, e2)
    d_est = estimate_nn_distance(surface_atoms, i_center, e1, e2)

    L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

    cell = surface_atoms.get_cell()
    rep_pos3d, rep_sym, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)

    rep_pos2d = project_positions_to_plane(rep_pos3d, origin_plane, e1, e2)
    base_pos2d = project_positions_to_plane(surface_atoms.positions, origin_plane, e1, e2)

    central_tile = np.array([MAX_REPEAT // 2, MAX_REPEAT // 2])
    central_shift3d = central_tile[0] * cell[0] + central_tile[1] * cell[1]
    shift_central2d = project_vector_to_plane_2d(central_shift3d, e1, e2)

    center2d = base_pos2d[i_center] + shift_central2d
    x0, y0 = center2d - 0.5 * L
    dx = 0.5 * (L - PLOT_L_VIEW)
    x0v, y0v = x0 + dx, y0 + dx

    window_view = box(0, 0, PLOT_L_VIEW, PLOT_L_VIEW)

    a1_2d = project_vector_to_plane_2d(cell[0], e1, e2)
    a1_norm = np.linalg.norm(a1_2d)
    if a1_norm < EPS:
        n_tile = 3
    else:
        n_tile = int(np.ceil(PLOT_L_VIEW / a1_norm)) + 3

    try:
        vor = Voronoi(rep_pos2d)
    except QhullError as exc:
        raise RuntimeError(f"Voronoi construction failed: {exc}") from exc

    central_mask = np.all(tile_id == central_tile, axis=1)

    central_polys = []
    poly_vals = []
    poly_local_indices = []

    for idx in np.where(central_mask)[0]:
        reg = vor.regions[vor.point_region[idx]]

        if -1 in reg or len(reg) < 3:
            continue

        poly = order_polygon_ccw(vor.vertices[reg])
        local = idx % n_surf

        Aref = area_ref_by_symbol.get(surface_syms[local])
        if Aref is None or not np.isfinite(Aref) or Aref <= 0.0:
            continue

        # Voronoi is already constructed in the fitted surface plane.
        # No A2d / |n_z| correction is applied. update from previous version
        A = polygon_area(poly)
        central_polys.append(poly)
        poly_local_indices.append(local)
        poly_vals.append(100.0 * (A - Aref) / Aref)

    if len(central_polys) == 0:
        raise RuntimeError("No finite central Voronoi polygons were produced.")

    poly_vals = np.asarray(poly_vals, dtype=float)

    bonds = build_bridge_bonds(surface_atoms, e1, e2)

    if len(bonds) == 0:
        raise RuntimeError("No bonds were produced.")

    bond_lines = []
    bond_vals = []

    for ix in range(-n_tile, n_tile + 1):
        for iy in range(-n_tile, n_tile + 1):
            shift3d = ix * cell[0] + iy * cell[1]
            shift2d = project_vector_to_plane_2d(shift3d, e1, e2)

            for b in bonds:
                p0 = base_pos2d[b["i"]] + shift_central2d + shift2d - np.array([x0v, y0v])
                p1 = p0 + project_vector_to_plane_2d(b["vec_ab"], e1, e2)

                if (
                    (0 <= p0[0] <= PLOT_L_VIEW and 0 <= p0[1] <= PLOT_L_VIEW)
                    or (0 <= p1[0] <= PLOT_L_VIEW and 0 <= p1[1] <= PLOT_L_VIEW)
                ):
                    bond_lines.append([p0, p1])
                    bond_vals.append(b["pct_dev"])

    bond_vals = np.asarray(bond_vals, dtype=float)

    # ========================================================
    # CSV DATA COLLECTION
    # ========================================================

    au_au = []
    au_cu = []
    cu_cu = []

    for b in bonds:
        s1 = surface_syms[b["i"]]
        s2 = surface_syms[b["j"]]
        pair = tuple(sorted((s1, s2)))

        if pair == ("Au", "Au"):
            au_au.append(b["pct_dev"])
        elif pair == ("Au", "Cu"):
            au_cu.append(b["pct_dev"])
        elif pair == ("Cu", "Cu"):
            cu_cu.append(b["pct_dev"])

    au_au_avg = mean_or_nan(au_au)
    au_cu_avg = mean_or_nan(au_cu)
    cu_cu_avg = mean_or_nan(cu_cu)

    au_area = []
    cu_area = []

    for val, local in zip(poly_vals, poly_local_indices):
        sym = surface_syms[local]
        if sym == "Au":
            au_area.append(val)
        elif sym == "Cu":
            cu_area.append(val)

    au_avg_area = mean_or_nan(au_area)
    cu_avg_area = mean_or_nan(cu_area)
    avg_au_cu_area = mean_or_nan(au_area + cu_area)

    diff_area = au_avg_area - cu_avg_area if len(au_area) and len(cu_area) else float("nan")

    # ========================================================
    # PLOTTING
    # ========================================================

    fig, ax = plt.subplots(figsize=(6, 6))

    offset_view = np.array([x0v, y0v])

    for poly, val in zip(central_polys, poly_vals):
        for ix in range(-n_tile, n_tile + 1):
            for iy in range(-n_tile, n_tile + 1):
                shift3d = ix * cell[0] + iy * cell[1]
                shift2d = project_vector_to_plane_2d(shift3d, e1, e2)
                poly2d = poly + shift2d - offset_view

                clipped = SPoly(poly2d).intersection(window_view)

                for clipped_poly in iter_polygons(clipped):
                    if clipped_poly.area <= 0:
                        continue
                    ax.add_patch(
                        Polygon(
                            np.asarray(clipped_poly.exterior.coords),
                            facecolor=CMAP(norm(val)),
                            linewidth=0,
                            zorder=1,
                        )
                    )

    if bond_lines:
        ax.add_collection(
            LineCollection(
                bond_lines,
                colors=BOND_OUTLINE_COLOR,
                linewidths=BOND_WIDTH_OUTLINE,
                zorder=3,
                capstyle="round",
                joinstyle="round",
            )
        )

        lc = LineCollection(
            bond_lines,
            cmap=CMAP,
            norm=norm,
            linewidths=BOND_WIDTH_MAIN,
            zorder=4,
            capstyle="round",
            joinstyle="round",
        )
        lc.set_array(bond_vals)
        ax.add_collection(lc)

    rep2d_for_plot = rep_pos2d - offset_view
    mask = (
        (rep2d_for_plot[:, 0] >= 0)
        & (rep2d_for_plot[:, 0] <= PLOT_L_VIEW)
        & (rep2d_for_plot[:, 1] >= 0)
        & (rep2d_for_plot[:, 1] <= PLOT_L_VIEW)
    )

    ax.scatter(
        rep2d_for_plot[mask, 0],
        rep2d_for_plot[mask, 1],
        c=[ATOM_COLORS.get(s, "gray") for s in np.asarray(rep_sym)[mask]],
        s=ATOM_SIZE,
        edgecolors="black",
        zorder=5,
    )

    # Adsorbate markers only: OH oxygen is shown as an x; bare O is shown as a triangle.
    # Adsorbates are assumed to use ASE's usual adsorbate tag 0.
    ads_indices = np.where(tags == 0)[0]
    if len(ads_indices) > 0:
        syms_all = np.asarray(atoms.get_chemical_symbols())
        ads_syms = syms_all[ads_indices]
        O_indices = ads_indices[ads_syms == "O"]
        H_indices = ads_indices[ads_syms == "H"]

        OH_O = []
        O_only = []

        for oi in O_indices:
            is_OH = False
            for hi in H_indices:
                vec_oh, _ = find_mic(
                    atoms.positions[hi] - atoms.positions[oi],
                    atoms.get_cell(),
                    atoms.get_pbc(),
                )
                if np.linalg.norm(vec_oh) < 1.2:
                    is_OH = True
                    break

            if is_OH:
                OH_O.append(oi)
            else:
                O_only.append(oi)

        for idx_list, marker in [(OH_O, "x"), (O_only, "^")]:
            if not idx_list:
                continue

            pts2d_all = []
            ads_base_pos = atoms.positions[idx_list]

            for ix in range(-n_tile, n_tile + 1):
                for iy in range(-n_tile, n_tile + 1):
                    shift3d = central_shift3d + ix * cell[0] + iy * cell[1]
                    pts2d = project_positions_to_plane(
                        ads_base_pos + shift3d,
                        origin_plane,
                        e1,
                        e2,
                    ) - offset_view
                    pts2d_all.append(pts2d)

            pts2d_all = np.vstack(pts2d_all)
            ads_mask = (
                (pts2d_all[:, 0] >= 0)
                & (pts2d_all[:, 0] <= PLOT_L_VIEW)
                & (pts2d_all[:, 1] >= 0)
                & (pts2d_all[:, 1] <= PLOT_L_VIEW)
            )

            if not np.any(ads_mask):
                continue

            if marker == "x":
                ax.scatter(
                    pts2d_all[ads_mask, 0],
                    pts2d_all[ads_mask, 1],
                    marker="x",
                    c="black",
                    s=ATOM_SIZE * 1.4,
                    linewidths=2.5,
                    zorder=8,
                )
            else:
                ax.scatter(
                    pts2d_all[ads_mask, 0],
                    pts2d_all[ads_mask, 1],
                    marker="^",
                    facecolors="black",
                    edgecolors="black",
                    s=ATOM_SIZE * 1.2,
                    linewidths=1.5,
                    zorder=8,
                )

    ax.set_xlim(0, PLOT_L_VIEW)
    ax.set_ylim(0, PLOT_L_VIEW)
    ax.set_title(
        f"Surface Au {au_surf:.0f}% | Subsurface Au {au_sub:.0f}%",
        fontsize=12,
    )
    ax.set_aspect("equal")

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        extend="both",
        shrink=0.85,
        aspect=35,
        pad=0.02,
    )

    major_ticks = [-10, -5, -2.5, 0, 2.5, 5, 10]
    cbar.set_ticks(major_ticks)
    cbar.set_ticklabels(["≤ −10", "−5", "−2.5", "0", "2.5", "5.0", "≥ 10"])

    minor_ticks = [-3.75, -1.25, 1.25, 3.75]
    cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    cbar.ax.tick_params(which="major", length=8, width=1.3)
    cbar.ax.tick_params(which="minor", length=4, width=0.8)
    cbar.set_label("Strain (%)", rotation=90, labelpad=12)

    plt.tight_layout()
    plot_path = OUTDIR / f"strain_row_{row.id}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    diagnostics = {
        "n_plane": n_plane,
        "plane_rms_A": plane_rms_A,
        "plane_max_abs_A": plane_max_abs_A,
        "central_points": int(np.sum(central_mask)),
        "polys_plotted": len(central_polys),
        "bonds_plotted": len(bond_lines),
        "unique_bonds": len(bonds),
        "plot_path": plot_path,
    }

    summary_row = [
        row.id,
        au_surf,
        au_sub,
        plane_rms_A,
        plane_max_abs_A,
        au_au_avg,
        au_cu_avg,
        cu_cu_avg,
        au_avg_area,
        cu_avg_area,
        avg_au_cu_area,
        diff_area,
    ]

    return summary_row, diagnostics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_file(DB_PATH)

    area_ref_by_symbol = {
        "Cu": ref_voronoi_area_from_bond_length("Cu"),
        "Au": ref_voronoi_area_from_bond_length("Au"),
    }

    norm = SymLogNorm(
        linthresh=CB_LINTHRESH,
        linscale=CB_LINSCALE,
        vmin=-CB_VMAX,
        vmax=CB_VMAX,
        base=10,
    )

    rows_processed = 0
    rows_skipped = 0

    with connect(DB_PATH) as db:
        rows = select_rows(db, ROW_IDS)

        with open(CSV_PATH, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "row_id",
                    "surface_au_composition",
                    "substrate_au_composition",
                    "plane_rms_A",
                    "plane_max_abs_A",
                    "au_au_avg_bond_strain_surface",
                    "au_cu_avg_bond_strain_surface",
                    "cu_cu_avg_bond_strain_surface",
                    "au_avg_voronoi_strain",
                    "cu_avg_voronoi_strain",
                    "avg_au_cu_area_strain",
                    "diff_area_au_minus_cu_strain",
                ]
            )
            csvfile.flush()
            os.fsync(csvfile.fileno())

            for idx, row in enumerate(rows):
                print(f"[{idx + 1}/{len(rows)}] Processing row_id={row.id}")

                try:
                    summary_row, diagnostics = analyze_and_plot_row(row, area_ref_by_symbol, norm)
                    writer.writerow(summary_row)
                    csvfile.flush()
                    os.fsync(csvfile.fileno())

                    rows_processed += 1

                    n_plane = diagnostics["n_plane"]
                    print(f"[saved] row_id={row.id}")
                    print(f"        n_plane={n_plane}")
                    print(
                        f"        plane RMS={diagnostics['plane_rms_A']:.4f} Å, "
                        f"max |deviation|={diagnostics['plane_max_abs_A']:.4f} Å"
                    )
                    print(
                        "        central points={central_points}, polys={polys_plotted}, "
                        "bonds plotted={bonds_plotted}, unique bonds={unique_bonds}".format(
                            **diagnostics
                        )
                    )
                    print(f"        plot={diagnostics['plot_path']}")

                except Exception as exc:
                    rows_skipped += 1
                    print(f"[skipped] row_id={row.id}")
                    print(f"        reason: {exc}")

    print()
    print("=" * 56)
    print("Run summary")
    print("=" * 56)
    print(f"Rows processed: {rows_processed}")
    print(f"Rows skipped  : {rows_skipped}")
    print(f"All strain plots saved to: {OUTDIR}")
    print(f"CSV summary saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
