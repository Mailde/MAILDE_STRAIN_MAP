"""
Atomic-scale projected trimer strain mapping for Au/Cu slab ASE databases.

This version is aligned with the Code 2 geometry convention:
- bond strain and trimer-area strain are both computed in the same fitted surface-plane basis;
- plotting also uses the same fitted surface-plane coordinates;
- no lattice-XY projection is mixed into the bond/trimer geometry.

Different target from Code 2:
- Code 2 computes atom-centred Voronoi area strain.
- This script computes triangle/trimer area strain from nearest-neighbour surface trimers.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic

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
    CMAP = plt.get_cmap("coolwarm")


# ============================================================
# USER SETTINGS
# ============================================================

DB_PATH = "example.db"

# Examples:
# ROW_IDS = "all"
# ROW_IDS = [4831, 4832, 4833]
# ROW_IDS = range(4831, 4900)
ROW_IDS = range(0, 200)

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

OUTDIR = Path("strain_ads_trimer")
OUTDIR.mkdir(exist_ok=True)
CSV_PATH = OUTDIR / "strain_summary_trimers.csv"

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

    if row_ids == "all" or row_ids is None:
        rows = all_rows
    elif isinstance(row_ids, list):
        if not all(isinstance(i, int) for i in row_ids):
            raise TypeError("All ROW_IDS list elements must be integers.")
        valid_ids = set(row_ids)
        rows = [row for row in all_rows if row.id in valid_ids]
    elif isinstance(row_ids, range):
        rows = [row for row in all_rows if row.id in row_ids]
    else:
        raise ValueError("ROW_IDS must be 'all', None, list[int], or range.")

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

    if n[2] < 0:
        n = -n

    return n


def plane_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def project_positions_to_plane(
    pos3d: np.ndarray,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
) -> np.ndarray:
    pos3d = np.asarray(pos3d, dtype=float)
    r = pos3d - np.asarray(origin, dtype=float)
    return np.column_stack((r @ e1, r @ e2))


def project_vector_to_plane_2d(vec3d: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    vec3d = np.asarray(vec3d, dtype=float)
    return np.array([vec3d @ e1, vec3d @ e2], dtype=float)


def plane_deviation_stats(pos3d: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    pos3d = np.asarray(pos3d, dtype=float)
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)

    nn = np.linalg.norm(normal)
    if nn < 1e-12:
        raise ValueError("Cannot compute plane deviation from near-zero normal.")

    normal = normal / nn
    d = (pos3d - origin) @ normal

    return float(np.sqrt(np.mean(d**2))), float(np.max(np.abs(d)))


# ============================================================
# SURFACE REPLICATION AND GEOMETRY
# ============================================================

def repeat_surface(surface_atoms, nx: int, ny: int):
    pos = np.asarray(surface_atoms.positions, dtype=float)
    cell = np.asarray(surface_atoms.get_cell(), dtype=float)
    syms = surface_atoms.get_chemical_symbols()

    all_pos = []
    all_sym = []
    tile_id = []
    orig_idx = []

    for ix in range(nx):
        for iy in range(ny):
            shift = ix * cell[0] + iy * cell[1]
            all_pos.append(pos + shift)
            all_sym.extend(syms)
            tile_id.extend([(ix, iy)] * len(pos))
            orig_idx.extend(range(len(pos)))

    return (
        np.vstack(all_pos),
        all_sym,
        np.asarray(tile_id, dtype=int),
        np.asarray(orig_idx, dtype=int),
    )


def find_central_surface_atom(surface_atoms, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> int:
    p2d = project_positions_to_plane(surface_atoms.positions, origin, e1, e2)
    c = p2d.mean(axis=0)
    return int(np.argmin(np.sum((p2d - c) ** 2, axis=1)))


def triangle_area_2d(p0, p1, p2) -> float:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    v1 = p1 - p0
    v2 = p2 - p0

    return 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])


def heron_area(a: float, b: float, c: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return float("nan")

    s = 0.5 * (a + b + c)
    val = s * (s - a) * (s - b) * (s - c)

    if val <= 0:
        return float("nan")

    return math.sqrt(val)


def trimer_reference_area(symbols3: list[str]) -> float:
    s0, s1, s2 = symbols3

    d01 = ref_bond_length(s0, s1)
    d12 = ref_bond_length(s1, s2)
    d20 = ref_bond_length(s2, s0)

    return heron_area(d01, d12, d20)


def trimer_label(symbols3: list[str]) -> str:
    n_au = sum(s == "Au" for s in symbols3)
    n_cu = 3 - n_au

    if n_au == 3:
        return "Au3"
    if n_cu == 3:
        return "Cu3"
    if n_au == 2:
        return "Au2Cu"
    return "AuCu2"


def iter_polygons(geom: BaseGeometry):
    if geom.is_empty:
        return

    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for g in geom.geoms:
            yield from iter_polygons(g)


def add_clipped_polygon(ax, poly_xy, window_poly, facecolor, zorder=1):
    poly_xy = np.asarray(poly_xy, dtype=float)

    if poly_xy.shape != (3, 2):
        raise ValueError(f"Expected triangle with shape (3, 2), got {poly_xy.shape}")

    tri = SPoly(poly_xy)

    if not tri.is_valid:
        tri = tri.buffer(0)

    clipped = tri.intersection(window_poly)

    for g in iter_polygons(clipped):
        if g.is_empty or g.area <= 0:
            continue

        coords = np.asarray(g.exterior.coords, dtype=float)

        if len(coords) < 3:
            continue

        ax.add_patch(
            Polygon(
                coords,
                facecolor=facecolor,
                linewidth=0,
                zorder=zorder,
            )
        )


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
# TRIMERS
# ============================================================

def build_surface_trimers(
    surface_atoms,
    origin: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
):
    """
    Build unique trimers from the projected nearest-neighbour bond graph.

    A trimer is accepted only if all three pair edges are present in the
    nearest-neighbour graph. The trimer area is computed directly in the same
    fitted surface-plane basis used for bond lengths.
    """
    syms = surface_atoms.get_chemical_symbols()
    n = len(surface_atoms)

    bonds = build_bridge_bonds(surface_atoms, e1, e2)
    edge_set = {tuple(sorted((b["i"], b["j"]))) for b in bonds}

    neighbors = {i: set() for i in range(n)}

    for i, j in edge_set:
        neighbors[i].add(j)
        neighbors[j].add(i)

    base2d = project_positions_to_plane(surface_atoms.positions, origin, e1, e2)

    trimers = {}

    for i in range(n):
        neigh = sorted(neighbors[i])

        for j, k in itertools.combinations(neigh, 2):
            if tuple(sorted((j, k))) not in edge_set:
                continue

            key = tuple(sorted((i, j, k)))

            if key in trimers:
                continue

            p_i = base2d[i]
            p_j = p_i + project_vector_to_plane_2d(mic_vec(surface_atoms, i, j), e1, e2)
            p_k = p_i + project_vector_to_plane_2d(mic_vec(surface_atoms, i, k), e1, e2)

            poly2d = np.vstack([p_i, p_j, p_k])
            area = triangle_area_2d(poly2d[0], poly2d[1], poly2d[2])

            if area < 1e-8:
                continue

            symbols3 = [syms[i], syms[j], syms[k]]
            area_ref = trimer_reference_area(symbols3)

            if not np.isfinite(area_ref) or area_ref < 1e-12:
                continue

            trimers[key] = {
                "key": key,
                "verts": (i, j, k),
                "poly2d": poly2d,
                "symbols": symbols3,
                "label": trimer_label(symbols3),
                "area": area,
                "area_ref": area_ref,
                "pct_dev": 100.0 * (area - area_ref) / area_ref,
            }

    return list(trimers.values()), bonds, base2d


# ============================================================
# ROW ANALYSIS
# ============================================================

def analyze_and_plot_row(row, norm: SymLogNorm):
    atoms = row.toatoms()

    au_surf, au_sub = au_composition(atoms, SURFACE_TAG)

    tags = np.asarray(atoms.get_tags())
    surface_atoms = atoms[tags == SURFACE_TAG]

    if len(surface_atoms) < 3:
        raise RuntimeError(f"Need at least 3 surface atoms; found {len(surface_atoms)}.")

    cell = np.asarray(surface_atoms.get_cell(), dtype=float)

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

    rep_pos3d, rep_sym, tile_id, orig_idx = repeat_surface(
        surface_atoms,
        MAX_REPEAT,
        MAX_REPEAT,
    )

    rep_pos2d = project_positions_to_plane(rep_pos3d, origin_plane, e1, e2)
    base_pos2d = project_positions_to_plane(surface_atoms.positions, origin_plane, e1, e2)

    central_tile = np.array([MAX_REPEAT // 2, MAX_REPEAT // 2])
    central_shift3d = central_tile[0] * cell[0] + central_tile[1] * cell[1]
    shift_central2d = project_vector_to_plane_2d(central_shift3d, e1, e2)

    center2d = base_pos2d[i_center] + shift_central2d
    x0, y0 = center2d - 0.5 * L
    dx = 0.5 * (L - PLOT_L_VIEW)
    x0v, y0v = x0 + dx, y0 + dx
    offset_view = np.array([x0v, y0v])

    window_view = box(0.0, 0.0, PLOT_L_VIEW, PLOT_L_VIEW)

    a1_2d = project_vector_to_plane_2d(cell[0], e1, e2)
    a2_2d = project_vector_to_plane_2d(cell[1], e1, e2)
    tile_scale = max(np.linalg.norm(a1_2d), np.linalg.norm(a2_2d), EPS)
    n_tile = int(np.ceil(PLOT_L_VIEW / tile_scale)) + 3

    trimers, bonds, base2d = build_surface_trimers(
        surface_atoms,
        origin_plane,
        e1,
        e2,
    )

    if len(trimers) == 0:
        raise RuntimeError("No valid surface trimers were produced.")

    if len(bonds) == 0:
        raise RuntimeError("No valid projected bonds were produced.")

    print(f"row {row.id}: {len(bonds)} bonds, {len(trimers)} trimers")

    # --------------------------------------------------------
    # Bond lines for overlay, in the same fitted surface plane
    # --------------------------------------------------------

    bond_lines = []
    bond_vals = []

    for ix in range(-n_tile, n_tile + 1):
        for iy in range(-n_tile, n_tile + 1):
            shift3d = ix * cell[0] + iy * cell[1]
            shift2d = project_vector_to_plane_2d(shift3d, e1, e2)

            for b in bonds:
                p0 = base2d[b["i"]] + shift_central2d + shift2d - offset_view
                p1 = p0 + project_vector_to_plane_2d(b["vec_ab"], e1, e2)

                inside0 = (0 <= p0[0] <= PLOT_L_VIEW) and (0 <= p0[1] <= PLOT_L_VIEW)
                inside1 = (0 <= p1[0] <= PLOT_L_VIEW) and (0 <= p1[1] <= PLOT_L_VIEW)

                if inside0 or inside1:
                    bond_lines.append([p0, p1])
                    bond_vals.append(b["pct_dev"])

    bond_vals = np.asarray(bond_vals, dtype=float)

    # --------------------------------------------------------
    # CSV data collection
    # --------------------------------------------------------

    surface_syms = surface_atoms.get_chemical_symbols()

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

    tri_au3 = [t["pct_dev"] for t in trimers if t["label"] == "Au3"]
    tri_au2cu = [t["pct_dev"] for t in trimers if t["label"] == "Au2Cu"]
    tri_aucu2 = [t["pct_dev"] for t in trimers if t["label"] == "AuCu2"]
    tri_cu3 = [t["pct_dev"] for t in trimers if t["label"] == "Cu3"]
    tri_all = [t["pct_dev"] for t in trimers]

    summary_row = [
        row.id,
        au_surf,
        au_sub,
        plane_rms_A,
        plane_max_abs_A,
        mean_or_nan(au_au),
        mean_or_nan(au_cu),
        mean_or_nan(cu_cu),
        mean_or_nan(tri_au3),
        mean_or_nan(tri_au2cu),
        mean_or_nan(tri_aucu2),
        mean_or_nan(tri_cu3),
        mean_or_nan(tri_all),
    ]

    # --------------------------------------------------------
    # Plotting projected trimer strain in same plane basis
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6))

    for tr in trimers:
        tri0 = tr["poly2d"]

        for ix in range(-n_tile, n_tile + 1):
            for iy in range(-n_tile, n_tile + 1):
                shift3d = ix * cell[0] + iy * cell[1]
                shift2d = project_vector_to_plane_2d(shift3d, e1, e2)

                poly_xy = tri0 + shift_central2d + shift2d - offset_view

                add_clipped_polygon(
                    ax=ax,
                    poly_xy=poly_xy,
                    window_poly=window_view,
                    facecolor=CMAP(norm(tr["pct_dev"])),
                    zorder=1,
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


    # --------------------------------------------------------
    # Adsorbate markers in the same trimer projection
    # --------------------------------------------------------
    # Convention copied from Code 2:
    # - O atom belonging to OH: marker "x"
    # - bare O atom: marker "^"
    # Adsorbates are assumed to have ASE tag 0.
    ads_markers_plotted = 0
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

            ads_markers_plotted += int(np.sum(ads_mask))

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
    cbar.set_label("Projected trimer strain (%)", rotation=90, labelpad=12)

    plt.tight_layout()
    plot_path = OUTDIR / f"trimer_strain_row_{row.id}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)

    diagnostics = {
        "n_plane": n_plane,
        "plane_rms_A": plane_rms_A,
        "plane_max_abs_A": plane_max_abs_A,
        "n_bonds": len(bonds),
        "n_trimers": len(trimers),
        "bonds_plotted": len(bond_lines),
        "ads_markers_plotted": ads_markers_plotted,
        "plot_path": plot_path,
    }

    return summary_row, diagnostics


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_file(DB_PATH)

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
                    "subsurface_au_composition",
                    "plane_rms_A",
                    "plane_max_abs_A",
                    "au_au_avg_bond_strain_surface",
                    "au_cu_avg_bond_strain_surface",
                    "cu_cu_avg_bond_strain_surface",
                    "au3_avg_trimer_strain",
                    "au2cu_avg_trimer_strain",
                    "aucu2_avg_trimer_strain",
                    "cu3_avg_trimer_strain",
                    "all_trimers_avg_strain",
                ]
            )
            csvfile.flush()
            os.fsync(csvfile.fileno())

            for idx, row in enumerate(rows):
                print(f"[{idx + 1}/{len(rows)}] Processing row_id={row.id}")

                try:
                    summary_row, diagnostics = analyze_and_plot_row(row, norm)

                    writer.writerow(summary_row)
                    csvfile.flush()
                    os.fsync(csvfile.fileno())

                    rows_processed += 1

                    print(f"[saved] row_id={row.id}")
                    print(f"        n_plane={diagnostics['n_plane']}")
                    print(
                        f"        plane RMS={diagnostics['plane_rms_A']:.4f} Å, "
                        f"max |deviation|={diagnostics['plane_max_abs_A']:.4f} Å"
                    )
                    print(
                        f"        bonds={diagnostics['n_bonds']}, "
                        f"trimers={diagnostics['n_trimers']}, "
                        f"bonds plotted={diagnostics['bonds_plotted']}, "
                        f"ads markers plotted={diagnostics['ads_markers_plotted']}"
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
    print(f"All trimer strain plots saved to: {OUTDIR}")
    print(f"CSV summary saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
