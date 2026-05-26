import os
import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic
from scipy.spatial import Voronoi

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable

from shapely.geometry import Polygon as SPoly, box
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator
from cmcrameri import cm

# ================================
# USER SETTINGS
# ================================
DB_PATH = "test.db"
ROW_IDS = None
SURFACE_TAG = 1

# If you have more than one Au-free slab in the DB, set this to the desired reference row id.
# If left as None, the code will auto-pick the smallest row id among Au-free candidates and print them.
REFERENCE_ROW_ID = None

N_NEIGHBORS = 6
LINEWIDTH = 3.0
ATOM_SIZE = 120.0

# --- WS / physics window ---
PLOT_N_ATOMS = 7
PLOT_SCALE = 1.10
PLOT_L_OVERRIDE = None

# --- Visualization window (Å) ---
PLOT_L_VIEW = 15.0   # adjust freely

MAX_REPEAT = 12

ATOM_COLORS = {"Au": "orange", "Pd": "blue", "Cu": "brown"}

CMAP = cm.vik
EPS = 1e-8

OUTDIR = "strain"
os.makedirs(OUTDIR, exist_ok=True)

CB_VMAX = 20.0        # physical saturation limit (±20%)
CB_LINTHRESH = 5.0    # high-resolution linear region
CB_LINSCALE = 2.5     # expands 0–5% visually


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
# GLOBAL REFERENCE SELECTION
# ================================
def is_pure_non_au(atoms):
    syms = np.array(atoms.get_chemical_symbols())
    return not np.any(syms == "Au")


# ================================
# REFERENCE DISTANCES FROM REFERENCE SLAB (6 NN)
# ================================
def build_reference_bond_lengths(atoms_ref):
    """
    Returns: dict { (elemA, elemB): float_distance_Angstrom }
    Built from the reference slab surface (tag == SURFACE_TAG),
    averaging in-plane distances to the 6 nearest neighbors per atom.
    For a pure-element reference slab, this will typically yield only (X,X).
    """
    tags = np.array(atoms_ref.get_tags())
    surface_atoms = atoms_ref[tags == SURFACE_TAG]
    if len(surface_atoms) == 0:
        return {}

    syms = surface_atoms.get_chemical_symbols()
    bond_ref = {}

    for i in range(len(surface_atoms)):
        dists = []
        for j in range(len(surface_atoms)):
            if i == j:
                continue
            d = np.linalg.norm(mic_vec(surface_atoms, i, j)[:2])  # in-plane
            dists.append((d, j))

        dists.sort(key=lambda x: x[0])
        for d, j in dists[:6]:
            a, b = sorted((syms[i], syms[j]))
            bond_ref.setdefault((a, b), []).append(d)

    return {k: float(np.mean(v)) for k, v in bond_ref.items()}


def _host_xx_from_D_REF():
    """
    For the agreed strain definition:
    - X–X uses X–X(ref)
    - X–Au uses X–X(ref)
    - Au–Au uses X–X(ref)
    Since the reference slab is pure non-Au, D_REF should contain exactly one (X,X).
    We return that (X,X) key and its value.
    """
    for (a, b), v in D_REF.items():
        if a == b and a != "Au" and np.isfinite(v):
            return (a, a), v
    return None, np.nan


def ref_bond_length(a, b):
    """
    Agreed behavior:
    - X–X: use X–X reference from the pure reference slab
    - X–Au: also use X–X reference (host lattice)
    - Au–Au: also use X–X reference (host lattice)
    """
    a, b = sorted((a, b))

    host_key, host_d0 = _HOST_KEY, _HOST_D0
    if not np.isfinite(host_d0) or host_key is None:
        return np.nan

    # If the pair itself is the host X–X and exists, use it (same as host_d0).
    if (a, b) in D_REF:
        return D_REF[(a, b)]

    # Any bond involving Au uses host X–X reference
    if a == "Au" or b == "Au":
        return host_d0

    # For any other pair not in D_REF (unlikely in pure-element reference workflow), skip.
    return np.nan


# ================================
# WS REFERENCE AREAS (FROM REFERENCE SLAB)
# ================================
def compute_ws_reference_areas(atoms_ref):
    tags = np.array(atoms_ref.get_tags())
    surface_atoms = atoms_ref[tags == SURFACE_TAG]
    if len(surface_atoms) == 0:
        return {}

    d_est = np.median([
        np.linalg.norm(mic_vec(surface_atoms, 0, j)[:2])
        for j in range(1, min(7, len(surface_atoms)))
    ])
    L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

    i_center = find_central_surface_atom(surface_atoms)

    rep_pos3d, _, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)
    e1, e2 = surface_basis_from_positions(rep_pos3d)
    rep_pos2d = project_positions(rep_pos3d, e1, e2)

    base_pos2d = project_positions(surface_atoms.positions, e1, e2)
    cell = surface_atoms.get_cell()

    shift_central2d = project_positions(
        (((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]).reshape(1, 3)),
        e1, e2
    )[0]

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
        if SPoly(poly - np.array([x0, y0])).within(window):
            area_ref[idx % n_surf] = polygon_area(poly)

    return area_ref


# ================================
# BRIDGE BONDS (UNCHANGED LOGIC, BUT d0 NOW HOST-REFERENCED)
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
            if not np.isfinite(d0) or d0 < EPS:
                continue

            pct = 100.0 * (d - d0) / d0
            bond_dict[(a, b)] = {
                "i": a,  # start atom index in base cell
                "vec_ab": vec if a == i else -vec,  # MIC vector from a -> b
                "pct_dev": pct
            }

    return list(bond_dict.values())


# ================================
# MAIN
# ================================
with connect(DB_PATH) as db:
    rows = list(db.select()) if ROW_IDS is None else [db.get(id=int(i)) for i in ROW_IDS]

    # ---- choose the SINGLE global reference slab (Au-free everywhere) ----
    if REFERENCE_ROW_ID is not None:
        ref_row = db.get(id=int(REFERENCE_ROW_ID))
        atoms_ref = ref_row.toatoms()
        if not is_pure_non_au(atoms_ref):
            raise RuntimeError(f"REFERENCE_ROW_ID={REFERENCE_ROW_ID} contains Au; choose an Au-free slab.")
        ref_row_id = ref_row.id
        au_free_candidates = None
    else:
        au_free_candidates = []
        for r in rows:
            a = r.toatoms()
            if is_pure_non_au(a):
                au_free_candidates.append((r.id, a))

        if len(au_free_candidates) == 0:
            raise RuntimeError("No Au-free reference slab found (no slab with zero Au everywhere).")

        # auto-pick smallest row id for reproducibility, but print candidates
        au_free_candidates.sort(key=lambda x: x[0])
        ref_row_id, atoms_ref = au_free_candidates[0]

    print("Using global reference slab:", ref_row_id)
    if au_free_candidates is not None and len(au_free_candidates) > 1:
        print("Note: multiple Au-free candidates found. Auto-picked smallest id.")
        print("Au-free candidate row IDs:", [rid for rid, _ in au_free_candidates])
        print("Tip: set REFERENCE_ROW_ID to choose explicitly.\n")

    # ---- build reference WS areas ONCE ----
    area_ref_by_local = compute_ws_reference_areas(atoms_ref)
    if not area_ref_by_local:
        raise RuntimeError("Reference slab produced no WS areas (check SURFACE_TAG / geometry).")

    # ---- build reference bond lengths ONCE (from reference slab) ----
    D_REF = build_reference_bond_lengths(atoms_ref)
    if not D_REF:
        raise RuntimeError("Reference slab produced no reference bond lengths (check SURFACE_TAG / geometry).")

    # Cache the host X–X reference for fast lookup in ref_bond_length()
    _HOST_KEY, _HOST_D0 = _host_xx_from_D_REF()
    if _HOST_KEY is None or not np.isfinite(_HOST_D0):
        raise RuntimeError("Could not determine host X–X reference distance from D_REF.")

    print("Reference bond lengths (Å) from reference slab (6 NN, in-plane):")
    for k, v in D_REF.items():
        print(f"  {k[0]}–{k[1]}: {v:.4f}")
    print()

    skipped = []

    for row in rows:
        atoms = row.toatoms()
        tags = np.array(atoms.get_tags())
        surface_atoms = atoms[tags == SURFACE_TAG]
        if len(surface_atoms) == 0:
            continue

        # --- WS / physics window ---
        d_est = np.median([
            np.linalg.norm(mic_vec(surface_atoms, 0, j)[:2])
            for j in range(1, min(7, len(surface_atoms)))
        ])
        L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

        i_center = find_central_surface_atom(surface_atoms)

        rep_pos3d, rep_sym, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)
        e1, e2 = surface_basis_from_positions(rep_pos3d)
        rep_pos2d = project_positions(rep_pos3d, e1, e2)

        base_pos2d = project_positions(surface_atoms.positions, e1, e2)
        cell = surface_atoms.get_cell()

        # central tile shift in 2D (IMPORTANT for bonds + view origin)
        shift_central2d = project_positions(
            (((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]).reshape(1, 3)),
            e1, e2
        )[0]

        center2d = base_pos2d[i_center] + shift_central2d
        x0, y0 = center2d - 0.5 * L

        # --- VIEW window ---
        if PLOT_L_VIEW > L:
            raise ValueError("PLOT_L_VIEW must be smaller than WS window")

        dx = 0.5 * (L - PLOT_L_VIEW)
        x0v, y0v = x0 + dx, y0 + dx
        window_view = box(0, 0, PLOT_L_VIEW, PLOT_L_VIEW)

        # --- determine translation range (for filling view) ---
        a1_xy = project_positions(cell[0].reshape(1, 3), e1, e2)[0]
        a2_xy = project_positions(cell[1].reshape(1, 3), e1, e2)[0]
        cell_len = min(np.linalg.norm(a1_xy), np.linalg.norm(a2_xy))
        n_tile = int(np.ceil(PLOT_L_VIEW / cell_len)) + 3  # +3 for safety

        # --- Voronoi (central tile only) ---
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
            local = idx % n_surf
            Aref = area_ref_by_local.get(local)
            if Aref is None:
                continue
            A = polygon_area(poly)
            poly_vals.append(100.0 * (A - Aref) / Aref)
            central_polys.append(poly)

        poly_vals = np.array(poly_vals) if poly_vals else np.array([])

        # --- Bonds (host-referenced for X–X, X–Au, Au–Au) ---
        bonds = build_bridge_bonds(surface_atoms)
        bond_lines, bond_vals = [], []

        for ix in range(-n_tile, n_tile + 1):
            for iy in range(-n_tile, n_tile + 1):
                shift_ij2d = project_positions(
                    ((ix * cell[0] + iy * cell[1]).reshape(1, 3)),
                    e1, e2
                )[0]

                for b in bonds:
                    # START at base cell atom, moved to central tile, then translated by ix/iy
                    p0 = base_pos2d[b["i"]] + shift_central2d + shift_ij2d - np.array([x0v, y0v])
                    dv = project_vector(b["vec_ab"], e1, e2)
                    p1 = p0 + dv

                    # keep only bonds whose start is inside the view
                    if 0 <= p0[0] <= PLOT_L_VIEW and 0 <= p0[1] <= PLOT_L_VIEW:
                        bond_lines.append([p0, p1])
                        bond_vals.append(b["pct_dev"])

        bond_vals = np.array(bond_vals) if bond_vals else np.array([])

        # --- Shared normalization ---
        all_vals = []
        if poly_vals.size:
            all_vals.append(poly_vals)
        if bond_vals.size:
            all_vals.append(bond_vals)

        if all_vals:
            all_vals = np.concatenate(all_vals)
        else:
            all_vals = np.array([0.0])

        norm = SymLogNorm(
            linthresh=CB_LINTHRESH,
            linscale=CB_LINSCALE,
            vmin=-CB_VMAX,
            vmax=CB_VMAX,
            base=10
        )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw polygons with tiling so the view fully fills
        for poly, val in zip(central_polys, poly_vals):
            for ix in range(-n_tile, n_tile + 1):
                for iy in range(-n_tile, n_tile + 1):
                    shift_ij2d = project_positions(
                        ((ix * cell[0] + iy * cell[1]).reshape(1, 3)),
                        e1, e2
                    )[0]
                    poly2d = poly + shift_ij2d - np.array([x0v, y0v])
                    clipped = SPoly(poly2d).intersection(window_view)
                    if clipped.is_empty:
                        continue
                    ax.add_patch(
                        Polygon(
                            np.array(clipped.exterior.coords),
                            facecolor=CMAP(norm(val)),
                            linewidth=0,
                            zorder=1
                        )
                    )

        # Draw bonds (make sure they are on top)
        if bond_lines:
            lc = LineCollection(bond_lines, cmap=CMAP, norm=norm, linewidths=LINEWIDTH, zorder=3)
            lc.set_array(bond_vals)
            ax.add_collection(lc)

        # Draw atoms from repeated positions (already global, just shift by view origin)
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
            label="Strain (%)",
            extend="both",
            shrink=0.75,
            aspect=30
        )

        major_ticks = [-10, -5, -2.5, 0, 2.5, 5, 10]
        cbar.set_ticks(major_ticks)
        major_labels = ["≤ −10", "−5", "−2.5", "0", "2.5", "5.0", "≥ 10"]
        cbar.set_ticklabels(major_labels)

        minor_ticks = [-3.75, -1.25, 1.25, 3.75]
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

        cbar.ax.tick_params(which="major", length=8, width=1.2)
        cbar.ax.tick_params(which="minor", length=4, width=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"strain_row_{row.id}.png"), dpi=200)
        plt.close()

    if skipped:
        print("Skipped rows:", skipped)

print("All strain plots saved to:", OUTDIR)

