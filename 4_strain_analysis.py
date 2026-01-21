import os
import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic
from scipy.spatial import Voronoi

from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable

from shapely.geometry import Polygon as SPoly, box


# ================================
# USER SETTINGS
# ================================
DB_PATH = "test.db"
ROW_IDS = None
SURFACE_TAG = 1

N_NEIGHBORS = 6
LINEWIDTH = 3.0
ATOM_SIZE = 120.0

PLOT_N_ATOMS = 6
PLOT_SCALE = 1.10
PLOT_L_OVERRIDE = None

MAX_REPEAT = 12

ATOM_COLORS = {"Au": "orange", "Pd": "blue", "Cu": "brown"}

D_REF = {
    ("Au", "Au"): 2.980,
    ("Au", "Pd"): 2.8975,
    ("Pd", "Pd"): 2.815,
    ("Cu", "Cu"): 2.609,
    ("Au", "Cu"): 2.7945,
}

CMAP = plt.cm.cividis

OUTDIR = "strain"
os.makedirs(OUTDIR, exist_ok=True)


# ================================
# SAFE NORMALIZATION
# ================================
from matplotlib.colors import TwoSlopeNorm, Normalize

def safe_norm(vals, min_range):
    """
    Absolutely safe normalization.
    Uses TwoSlopeNorm only if data spans zero.
    Otherwise falls back to Normalize.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    # Empty or invalid → symmetric dummy scale
    if vals.size == 0:
        return Normalize(vmin=-min_range, vmax=min_range)

    vmin_data = vals.min()
    vmax_data = vals.max()

    # All zero or numerically flat
    if np.isclose(vmin_data, vmax_data):
        if np.isclose(vmin_data, 0.0):
            return Normalize(vmin=-min_range, vmax=min_range)
        else:
            pad = max(abs(vmin_data) * 0.1, min_range)
            return Normalize(vmin=vmin_data - pad, vmax=vmax_data + pad)

    # Data does NOT cross zero → no TwoSlopeNorm allowed
    if vmin_data >= 0.0 or vmax_data <= 0.0:
        pad = max(np.percentile(np.abs(vals), 99) * 0.05, min_range)
        return Normalize(vmin=vmin_data - pad, vmax=vmax_data + pad)

    # Data crosses zero → safe TwoSlopeNorm
    vmax = np.percentile(np.abs(vals), 99)
    vmax = max(vmax, min_range)

    # 🔒 enforce strict inequality
    eps = max(vmax * 1e-6, 1e-6)
    return TwoSlopeNorm(vmin=-vmax - eps, vcenter=0.0, vmax=vmax + eps)
# ================================
# GEOMETRY HELPERS
# ================================
def ref_bond_length(a, b):
    return D_REF.get(tuple(sorted((a, b))), np.nan)


def ws_reference_area(sym):
    d0 = D_REF.get((sym, sym), np.nan)
    if not np.isfinite(d0):
        return np.nan
    return 0.5 * np.sqrt(3.0) * d0 * d0


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


def surface_basis_from_positions(pos):
    xy = pos[:, :2] - pos[:, :2].mean(axis=0)
    cov = np.cov(xy.T)
    w, v = np.linalg.eigh(cov)
    e1, e2 = v[:, np.argsort(w)[::-1]]
    return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)


def project_positions(pos3d, e1, e2):
    return np.column_stack((pos3d[:, :2] @ e1, pos3d[:, :2] @ e2))


def project_vector(vec3d, e1, e2):
    return np.array([vec3d[:2] @ e1, vec3d[:2] @ e2])


def find_central_surface_atom(surface_atoms):
    xy = surface_atoms.positions[:, :2]
    c = xy.mean(axis=0)
    return int(np.argmin(np.sum((xy - c) ** 2, axis=1)))


def order_polygon_ccw(poly):
    poly = np.asarray(poly, float)
    c = poly.mean(axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ================================
# BRIDGE BONDS
# ================================
def build_bridge_bonds(surface_atoms):
    syms = surface_atoms.get_chemical_symbols()
    n = len(surface_atoms)
    bond_dict = {}

    for i in range(n):
        cand = []
        for j in range(n):
            if i == j:
                continue
            vec = mic_vec(surface_atoms, i, j)
            d = np.linalg.norm(vec[:2])
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
    rows = db.select() if ROW_IDS is None else [db.get(id=int(i)) for i in ROW_IDS]

    for row in rows:
        atoms = row.toatoms()
        tags = np.array(atoms.get_tags())
        surface_atoms = atoms[tags == SURFACE_TAG]
        if len(surface_atoms) == 0:
            continue

        d_est = np.median([
            np.linalg.norm(mic_vec(surface_atoms, 0, j)[:2])
            for j in range(1, min(7, len(surface_atoms)))
        ])
        L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

        i_center = find_central_surface_atom(surface_atoms)

        nx = ny = MAX_REPEAT
        rep_pos3d, rep_sym, tile_id = repeat_surface(surface_atoms, nx, ny)

        e1, e2 = surface_basis_from_positions(rep_pos3d)
        rep_pos2d = project_positions(rep_pos3d, e1, e2)

        base_pos2d = project_positions(surface_atoms.positions, e1, e2)
        cell = surface_atoms.get_cell()

        shift2d = project_positions(
            ((nx // 2) * cell[0] + (ny // 2) * cell[1]).reshape(1, 3), e1, e2
        )[0]

        center2d = base_pos2d[i_center] + shift2d
        x0, y0 = center2d - 0.5 * L

        # ----------------------------
        # Voronoi (central tile only)
        # ----------------------------
        vor = Voronoi(rep_pos2d)

        central_tile = np.array([nx // 2, ny // 2])
        central_mask = np.all(tile_id == central_tile, axis=1)

        ws_polys, ws_vals = [], []

        for i in np.where(central_mask)[0]:
            reg = vor.regions[vor.point_region[i]]
            if -1 in reg or len(reg) < 3:
                continue

            poly = order_polygon_ccw(vor.vertices[reg])
            A = polygon_area(poly)
            A0 = ws_reference_area(rep_sym[i])
            if not np.isfinite(A0):
                continue

            ws_polys.append(poly)
            ws_vals.append(100.0 * (A - A0) / A0)

        ws_vals = np.array(ws_vals)

        # ----------------------------
        # Bonds
        # ----------------------------
        bonds = build_bridge_bonds(surface_atoms)
        bond_lines, bond_vals = [], []

        for ix in range(nx):
            for iy in range(ny):
                shift3d = ix * cell[0] + iy * cell[1]
                shift2d = project_positions(shift3d.reshape(1, 3), e1, e2)[0]

                for b in bonds:
                    p0 = base_pos2d[b["i"]] + shift2d - np.array([x0, y0])
                    dv = project_vector(b["vec_ab"], e1, e2)
                    p1 = p0 + dv
                    if not (0 <= p0[0] <= L and 0 <= p0[1] <= L):
                        continue
                    bond_lines.append([p0, p1])
                    bond_vals.append(b["pct_dev"])

        bond_vals = np.array(bond_vals)

        # ----------------------------
        # SAFE NORMALIZATION
        # ----------------------------
   
        norm_ws = safe_norm(ws_vals, min_range=0.5)
        norm_bond = safe_norm(bond_vals, min_range=1.0)


        # ----------------------------
        # Plot
        # ----------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        window = box(0, 0, L, L)

        translations = []
        for ix in range(-2, 3):
            for iy in range(-2, 3):
                shift3d = ix * cell[0] + iy * cell[1]
                translations.append(
                    project_positions(shift3d.reshape(1, 3), e1, e2)[0]
                )

        # WS background
        for poly, val in zip(ws_polys, ws_vals):
            for t in translations:
                poly2d = poly + t - np.array([x0, y0])
                clipped = SPoly(poly2d).intersection(window)
                if clipped.is_empty:
                    continue
                ax.add_patch(
                    Polygon(
                        np.array(clipped.exterior.coords),
                        facecolor=CMAP(norm_ws(val)),
                        alpha=0.45,
                        linewidth=0
                    )
                )

        # Bonds
        if len(bond_lines) > 0:
            lc = LineCollection(
                bond_lines, cmap=CMAP, norm=norm_bond, linewidths=LINEWIDTH
            )
            lc.set_array(bond_vals)
            ax.add_collection(lc)

        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.set_title(f"Row {row.id} – WS area + bond strain")

        # Colorbars
        sm_ws = ScalarMappable(norm=norm_ws, cmap=CMAP)
        sm_ws.set_array([])
        cbar_ws = plt.colorbar(sm_ws, ax=ax, location="left", pad=0.07)
        cbar_ws.set_label("WS area strain (%)")

        sm_bond = ScalarMappable(norm=norm_bond, cmap=CMAP)
        sm_bond.set_array([])
        cbar_bond = plt.colorbar(sm_bond, ax=ax, location="right", pad=0.07)
        cbar_bond.set_label("Bond strain (%)")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"strain_row_{row.id}.png"), dpi=200)
        plt.close()

print("All strain plots saved to:", OUTDIR)

