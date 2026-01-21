import os
import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic
from scipy.spatial import Voronoi

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator

from shapely.geometry import Polygon as SPoly, box
from cmcrameri import cm


# ================================
# USER SETTINGS
# ================================
DB_PATH = "test.db"
ROW_IDS = None
SURFACE_TAG = 1
REFERENCE_ROW_ID = None

N_NEIGHBORS = 6
LINEWIDTH = 3.0
ATOM_SIZE = 120.0

# --- WS / physics window ---
PLOT_N_ATOMS = 10
PLOT_SCALE = 1.10
PLOT_L_OVERRIDE = None

# --- Visualization window (Å) ---
PLOT_L_VIEW = 15.0
MAX_REPEAT = 13

ATOM_COLORS = {"Au": "orange", "Pd": "blue", "Cu": "brown"}
CMAP = cm.vik
EPS = 1e-8

OUTDIR = "strain"
os.makedirs(OUTDIR, exist_ok=True)

# --- Colorbar ---
CB_VMAX = 20.0
CB_LINTHRESH = 5.0
CB_LINSCALE = 2.5

# --- Bond visualization tuning ---
BOND_WIDTH_MAIN = LINEWIDTH
BOND_WIDTH_OUTLINE = LINEWIDTH + 1.2
BOND_OUTLINE_COLOR = "black"

# --- NEW: robust plane-fit control ---
Z_TOL_PLANE = 1.0  # Å  (median-z filter tolerance for defining the surface plane)


# ================================
# GEOMETRY HELPERS
# ================================
def mic_vec(atoms, i, j):
    vec, _ = find_mic(
        atoms.positions[j] - atoms.positions[i],
        atoms.get_cell(),
        atoms.get_pbc()
    )
    return vec


def repeat_surface(surface_atoms, nx, ny):
    pos = surface_atoms.positions
    cell = surface_atoms.get_cell()
    syms = surface_atoms.get_chemical_symbols()

    all_pos, all_sym, tile_id = [], [], []
    for ix in range(nx):
        for iy in range(ny):
            shift = ix * cell[0] + iy * cell[1]
            all_pos.append(pos + shift)
            all_sym.extend(syms)
            tile_id.extend([(ix, iy)] * len(pos))

    return np.vstack(all_pos), all_sym, np.array(tile_id)


# -------------------------------
# NEW: best-fit surface plane (median-filtered) + lattice-based in-plane basis
# -------------------------------
def median_filtered_plane_normal(pos3d, z_tol=Z_TOL_PLANE):
    """
    Robust best-fit plane normal:
      1) keep points within z_tol of median(z) (reject outliers)
      2) fit plane via SVD; smallest singular vector is normal
    Returns unit normal n (shape (3,)).
    """
    if len(pos3d) < 3:
        return np.array([0.0, 0.0, 1.0])

    z = pos3d[:, 2]
    z0 = np.median(z)
    mask = np.abs(z - z0) < z_tol
    pts = pos3d[mask]
    if len(pts) < 3:
        pts = pos3d  # fallback

    origin = pts.mean(axis=0)
    X = pts - origin
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    n = VT[-1]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])

    n = n / n_norm
    return n


def project_vec_to_plane(v, n_unit):
    """v_parallel = v - (v·n) n"""
    return v - np.dot(v, n_unit) * n_unit


def lattice_inplane_basis(cell, n_plane):
    """
    Build an orthonormal in-plane basis (b1,b2) from lattice vectors a1,a2:
      - project a1 and a2 onto the fitted plane
      - Gram-Schmidt to get orthonormal b1,b2
    This removes PCA entirely and uses a physical lattice-derived frame.
    """
    a1 = np.array(cell[0], float)
    a2 = np.array(cell[1], float)

    u1 = project_vec_to_plane(a1, n_plane)
    u2 = project_vec_to_plane(a2, n_plane)

    n1 = np.linalg.norm(u1)
    if n1 < 1e-12:
        u1 = np.array([1.0, 0.0, 0.0])
        n1 = 1.0
    b1 = u1 / n1

    # orthogonalize u2 against b1
    u2_ortho = u2 - np.dot(u2, b1) * b1
    n2 = np.linalg.norm(u2_ortho)
    if n2 < 1e-12:
        # fallback: pick any vector in plane perpendicular to b1
        tmp = np.array([0.0, 0.0, 1.0])
        u2_ortho = np.cross(n_plane, b1)
        n2 = np.linalg.norm(u2_ortho)
        if n2 < 1e-12:
            u2_ortho = np.array([0.0, 1.0, 0.0])
            n2 = 1.0
    b2 = u2_ortho / n2

    return b1, b2


def project_positions(pos3d, b1, b2):
    """
    Project 3D positions into the in-plane orthonormal basis (b1,b2).
    Returns Nx2 coordinates in Å.
    """
    pos3d = np.asarray(pos3d, float)
    return np.column_stack((pos3d @ b1, pos3d @ b2))


def project_vector(vec3d, b1, b2):
    """
    Project 3D vector into (b1,b2) components.
    Returns length-2 array in Å.
    """
    vec3d = np.asarray(vec3d, float)
    return np.array([np.dot(vec3d, b1), np.dot(vec3d, b2)])


def find_central_surface_atom_from_2d(pos2d):
    c = pos2d.mean(axis=0)
    return int(np.argmin(np.sum((pos2d - c) ** 2, axis=1)))


def order_polygon_ccw(poly):
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ================================
# REFERENCE HELPERS
# ================================
def is_pure_non_au(atoms):
    syms = np.array(atoms.get_chemical_symbols())
    return not np.any(syms == "Au")


# ================================
# REFERENCE BOND LENGTHS (6 NN)
# ================================
def build_reference_bond_lengths(atoms_ref):
    tags = np.array(atoms_ref.get_tags())
    surface_atoms = atoms_ref[tags == SURFACE_TAG]

    # --- define plane + lattice basis for *reference* slab ---
    n_plane = median_filtered_plane_normal(surface_atoms.positions, z_tol=Z_TOL_PLANE)
    b1, b2 = lattice_inplane_basis(surface_atoms.get_cell(), n_plane)

    syms = surface_atoms.get_chemical_symbols()
    bond_ref = {}

    for i in range(len(surface_atoms)):
        dists = []
        for j in range(len(surface_atoms)):
            if i == j:
                continue
            v = mic_vec(surface_atoms, i, j)
            d = np.linalg.norm(project_vector(v, b1, b2))  # in-plane length (true)
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])

        for d, j in dists[:6]:
            a, b = sorted((syms[i], syms[j]))
            bond_ref.setdefault((a, b), []).append(d)

    return {k: float(np.mean(v)) for k, v in bond_ref.items()}


def _host_xx_from_D_REF():
    for (a, b), v in D_REF.items():
        if a == b and a != "Au":
            return (a, a), v
    return None, np.nan


def ref_bond_length(a, b):
    a, b = sorted((a, b))
    if a == "Au" or b == "Au":
        return _HOST_D0
    return D_REF.get((a, b), np.nan)


# ================================
# WS REFERENCE AREAS (TRUE SURFACE AREA)
# ================================
def compute_ws_reference_areas(atoms_ref):
    tags = np.array(atoms_ref.get_tags())
    surface_atoms = atoms_ref[tags == SURFACE_TAG]
    if len(surface_atoms) == 0:
        return {}

    # --- plane + lattice basis from this reference slab ---
    n_plane = median_filtered_plane_normal(surface_atoms.positions, z_tol=Z_TOL_PLANE)
    b1, b2 = lattice_inplane_basis(surface_atoms.get_cell(), n_plane)

    # estimate d_est using in-plane projected distances
    d_est = np.median([
        np.linalg.norm(project_vector(mic_vec(surface_atoms, 0, j), b1, b2))
        for j in range(1, min(7, len(surface_atoms)))
    ])
    L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

    # central atom in the same in-plane basis
    base_pos2d = project_positions(surface_atoms.positions, b1, b2)
    i_center = find_central_surface_atom_from_2d(base_pos2d)

    rep_pos3d, _, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)
    rep_pos2d = project_positions(rep_pos3d, b1, b2)
    cell = surface_atoms.get_cell()

    # shift corresponding to the central tile
    shift_central2d = project_vector(
        ((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]),
        b1, b2
    )

    center2d = base_pos2d[i_center] + shift_central2d
    x0, y0 = center2d - 0.5 * L
    window = box(0, 0, L, L)

    vor = Voronoi(rep_pos2d)
    central_tile = np.array([MAX_REPEAT // 2, MAX_REPEAT // 2])
    central_mask = np.all(tile_id == central_tile, axis=1)

    area_ref = {}
    n_surf = len(surface_atoms)

    for idx in np.where(central_mask)[0]:
        reg = vor.regions[vor.point_region[idx]]
        if -1 in reg or len(reg) < 3:
            continue
        poly = order_polygon_ccw(vor.vertices[reg])

        # poly is already in true surface-plane coordinates => polygon_area is TRUE surface area
        if SPoly(poly - np.array([x0, y0])).within(window):
            area_ref[idx % n_surf] = polygon_area(poly)

    return area_ref


# ================================
# BRIDGE BONDS
# ================================
def build_bridge_bonds(surface_atoms, b1, b2):
    """
    Same logic as your original, but:
      - distances are computed in lattice-derived in-plane coordinates (no PCA, no vec[:2])
    """
    syms = surface_atoms.get_chemical_symbols()
    n = len(surface_atoms)
    bond_dict = {}

    for i in range(n):
        cand = []
        for j in range(n):
            if i == j:
                continue
            vec = mic_vec(surface_atoms, i, j)
            d = np.linalg.norm(project_vector(vec, b1, b2))  # true in-plane distance
            cand.append((d, j, vec))
        cand.sort(key=lambda x: x[0])

        for d, j, vec in cand[:N_NEIGHBORS]:
            a, b = sorted((i, j))
            if (a, b) in bond_dict:
                continue

            d0 = ref_bond_length(syms[i], syms[j])
            if not np.isfinite(d0):
                continue

            pct = 100.0 * (d - d0) / d0
            bond_dict[(a, b)] = {
                "i": a,
                "vec_ab": vec if a == i else -vec,
                "pct_dev": pct
            }

    return list(bond_dict.values())


# ================================
# MAIN
# ================================
with connect(DB_PATH) as db:
    rows = list(db.select()) if ROW_IDS is None else [db.get(id=int(i)) for i in ROW_IDS]

    # --- global reference slab: first Au-free slab ---
    au_free = [(r.id, r.toatoms()) for r in rows if is_pure_non_au(r.toatoms())]
    au_free.sort(key=lambda x: x[0])
    ref_row_id, atoms_ref = au_free[0]

    print("Using global reference slab:", ref_row_id)

    # --- reference areas + bond lengths (consistent with this exact reference slab) ---
    area_ref_by_local = compute_ws_reference_areas(atoms_ref)
    D_REF = build_reference_bond_lengths(atoms_ref)
    _HOST_KEY, _HOST_D0 = _host_xx_from_D_REF()

    for row in rows:
        atoms = row.toatoms()
        tags = np.array(atoms.get_tags())
        surface_atoms = atoms[tags == SURFACE_TAG]
        if len(surface_atoms) == 0:
            continue

        # --- plane + lattice basis for THIS slab (no PCA) ---
        n_plane = median_filtered_plane_normal(surface_atoms.positions, z_tol=Z_TOL_PLANE)
        b1, b2 = lattice_inplane_basis(surface_atoms.get_cell(), n_plane)

        # --- geometry (same logic, but in-plane length uses b1,b2) ---
        d_est = np.median([
            np.linalg.norm(project_vector(mic_vec(surface_atoms, 0, j), b1, b2))
            for j in range(1, min(7, len(surface_atoms)))
        ])
        L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

        rep_pos3d, rep_sym, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)
        rep_pos2d = project_positions(rep_pos3d, b1, b2)
        base_pos2d = project_positions(surface_atoms.positions, b1, b2)
        i_center = find_central_surface_atom_from_2d(base_pos2d)
        cell = surface_atoms.get_cell()

        shift_central2d = project_vector(
            ((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]),
            b1, b2
        )

        center2d = base_pos2d[i_center] + shift_central2d
        x0, y0 = center2d - 0.5 * L
        dx = 0.5 * (L - PLOT_L_VIEW)
        x0v, y0v = x0 + dx, y0 + dx
        window_view = box(0, 0, PLOT_L_VIEW, PLOT_L_VIEW)

        a1_xy = project_vector(cell[0], b1, b2)
        n_tile = int(np.ceil(PLOT_L_VIEW / np.linalg.norm(a1_xy))) + 3

        # --- Voronoi (true surface-plane coordinates) ---
        vor = Voronoi(rep_pos2d)
        central_tile = np.array([MAX_REPEAT // 2, MAX_REPEAT // 2])
        central_mask = np.all(tile_id == central_tile, axis=1)

        central_polys, poly_vals = [], []
        n_surf = len(surface_atoms)

        for idx in np.where(central_mask)[0]:
            reg = vor.regions[vor.point_region[idx]]
            if -1 in reg or len(reg) < 3:
                continue
            poly = order_polygon_ccw(vor.vertices[reg])

            # same reference lookup as YOUR provided code
            Aref = area_ref_by_local.get(idx % n_surf)
            if Aref is None:
                continue

            # poly area is already TRUE surface area in the best-fit plane
            A = polygon_area(poly)

            central_polys.append(poly)
            poly_vals.append(100.0 * (A - Aref) / Aref)

        poly_vals = np.array(poly_vals)

        # --- Bonds ---
        bonds = build_bridge_bonds(surface_atoms, b1, b2)
        bond_lines, bond_vals = [], []

        for ix in range(-n_tile, n_tile + 1):
            for iy in range(-n_tile, n_tile + 1):
                shift = project_vector(ix * cell[0] + iy * cell[1], b1, b2)

                for b in bonds:
                    p0 = base_pos2d[b["i"]] + shift_central2d + shift - np.array([x0v, y0v])
                    p1 = p0 + project_vector(b["vec_ab"], b1, b2)

                    # --- FIX: draw bond if p0 OR p1 is inside the view ---
                    if (
                        (0 <= p0[0] <= PLOT_L_VIEW and 0 <= p0[1] <= PLOT_L_VIEW)
                        or
                        (0 <= p1[0] <= PLOT_L_VIEW and 0 <= p1[1] <= PLOT_L_VIEW)
                    ):
                        bond_lines.append([p0, p1])
                        bond_vals.append(b["pct_dev"])

        bond_vals = np.array(bond_vals)

        # --- normalization ---
        norm = SymLogNorm(
            linthresh=CB_LINTHRESH,
            linscale=CB_LINSCALE,
            vmin=-CB_VMAX,
            vmax=CB_VMAX,
            base=10
        )

        # --- PLOT ---
        fig, ax = plt.subplots(figsize=(6, 6))

        for poly, val in zip(central_polys, poly_vals):
            for ix in range(-n_tile, n_tile + 1):
                for iy in range(-n_tile, n_tile + 1):
                    shift = project_vector(ix * cell[0] + iy * cell[1], b1, b2)
                    poly2d = poly + shift - np.array([x0v, y0v])
                    clipped = SPoly(poly2d).intersection(window_view)
                    if not clipped.is_empty:
                        ax.add_patch(
                            Polygon(
                                np.array(clipped.exterior.coords),
                                facecolor=CMAP(norm(val)),
                                linewidth=0,
                                zorder=1
                            )
                        )

        # --- OPTIMIZED BONDS (NO ALPHA) ---
        if bond_lines:
            ax.add_collection(
                LineCollection(
                    bond_lines,
                    colors=BOND_OUTLINE_COLOR,
                    linewidths=BOND_WIDTH_OUTLINE,
                    zorder=3,
                    capstyle="round",
                    joinstyle="round"
                )
            )

            lc = LineCollection(
                bond_lines,
                cmap=CMAP,
                norm=norm,
                linewidths=BOND_WIDTH_MAIN,
                zorder=4,
                capstyle="round",
                joinstyle="round"
            )
            lc.set_array(bond_vals)
            ax.add_collection(lc)

        # --- atoms ---
        rep2d = rep_pos2d - np.array([x0v, y0v])
        mask = (
            (rep2d[:, 0] >= 0) & (rep2d[:, 0] <= PLOT_L_VIEW) &
            (rep2d[:, 1] >= 0) & (rep2d[:, 1] <= PLOT_L_VIEW)
        )

        ax.scatter(
            rep2d[mask, 0],
            rep2d[mask, 1],
            c=[ATOM_COLORS.get(s, "gray") for s in np.array(rep_sym)[mask]],
            s=ATOM_SIZE,
            edgecolors="black",
            zorder=5
        )

        ax.set_xlim(0, PLOT_L_VIEW)
        ax.set_ylim(0, PLOT_L_VIEW)
        ax.set_aspect("equal")
        ax.set_title(f"Row {row.id} – ref row {ref_row_id}")

        # --- COLORBAR ---
        sm = ScalarMappable(norm=norm, cmap=CMAP)
        sm.set_array([])

        cbar = plt.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            extend="both",
            shrink=0.85,
            aspect=35,
            pad=0.02
        )

        # Major ticks (exactly as in the reference image)
        major_ticks = [-10, -5, -2.5, 0, 2.5, 5, 10]
        cbar.set_ticks(major_ticks)
        cbar.set_ticklabels([
            "≤ −10",
            "−5",
            "−2.5",
            "0",
            "2.5",
            "5.0",
            "≥ 10"
        ])

        # Minor ticks (unlabeled)
        minor_ticks = [-7.5, -3.75, -1.25, 1.25, 3.75, 7.5]
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

        # Tick styling
        cbar.ax.tick_params(which="major", length=8, width=1.3)
        cbar.ax.tick_params(which="minor", length=4, width=0.8)

        # Label
        cbar.set_label("Strain (%)", rotation=90, labelpad=12)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"strain_row_{row.id}.png"), dpi=200)
        plt.close()

print("All strain plots saved to:", OUTDIR)

