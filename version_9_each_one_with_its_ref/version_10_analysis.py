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

N_NEIGHBORS = 6
LINEWIDTH = 5.0
ATOM_SIZE = 127.0

# --- WS / physics window ---
PLOT_N_ATOMS = 7
PLOT_SCALE = 1.10
PLOT_L_OVERRIDE = None

# --- Visualization window (Å) ---
PLOT_L_VIEW = 15.0
MAX_REPEAT = 12

ATOM_COLORS = {"Au": "orange", "Cu": "brown"}
CMAP = cm.vik
EPS = 1e-8

OUTDIR = "strain"
os.makedirs(OUTDIR, exist_ok=True)

# --- Reference bond lengths ---
D_REF = {
    ("Cu", "Cu"): 2.609,
    ("Au", "Au"): 2.980,
    ("Au", "Cu"): 2.7945,
}

# --- Colorbar / plotting ---
CB_VMAX = 20.0
CB_LINTHRESH = 5.0
CB_LINSCALE = 2.5

# --- Bond visualization (NO alpha) ---
BOND_WIDTH_MAIN = LINEWIDTH
BOND_WIDTH_OUTLINE = LINEWIDTH + 1.1
BOND_OUTLINE_COLOR = "black"

# --- Plane-fit control (median-z filter) ---
Z_TOL_PLANE = 0.57  # Å


# ================================
# GEOMETRY HELPERS
# ================================
def ref_bond_length(a, b):
    return D_REF.get(tuple(sorted((a, b))), np.nan)


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
# REPLACED: PCA -> lattice-vector coordinates (plot-safe)
# -------------------------------
def lattice_xy_basis_from_cell(cell):
    """
    Build a stable 2D orthonormal basis (b1,b2) from the in-plane lattice vectors.
    Uses only the XY components of cell[0], cell[1] (plotting coordinates remain 2D).
    """
    a1 = np.array(cell[0][:2], float)
    a2 = np.array(cell[1][:2], float)

    n1 = np.linalg.norm(a1)
    if n1 < 1e-12:
        # fallback: standard axes
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])

    b1 = a1 / n1

    # Gram-Schmidt for b2
    a2_perp = a2 - np.dot(a2, b1) * b1
    n2 = np.linalg.norm(a2_perp)
    if n2 < 1e-12:
        # if nearly collinear in xy, pick a perpendicular direction
        b2 = np.array([-b1[1], b1[0]])
    else:
        b2 = a2_perp / n2

    return b1, b2


def project_positions_lattice(pos3d, b1, b2):
    """
    Project 3D positions to 2D plot coordinates using the lattice-defined XY basis.
    """
    xy = pos3d[:, :2]
    return np.column_stack((xy @ b1, xy @ b2))


def project_vector_lattice(vec3d, b1, b2):
    """
    Project 3D vectors to 2D plot vectors using the lattice-defined XY basis.
    """
    vxy = vec3d[:2]
    return np.array([vxy @ b1, vxy @ b2])


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


def au_composition(atoms, surface_tag=1):
    tags = np.array(atoms.get_tags())
    syms = np.array(atoms.get_chemical_symbols())

    surf = syms[tags == surface_tag]
    subsurf = syms[tags == surface_tag + 1]

    def frac_au(arr):
        return 100.0 * np.sum(arr == "Au") / len(arr) if len(arr) else 0.0

    return frac_au(surf), frac_au(subsurf)


# -------------------------------
# Plane normal + projection onto plane (for bonds + area correction)
# -------------------------------
def median_filtered_plane_normal(pos3d, z_tol = Z_TOL_PLANE):
    """
    Robust best-fit plane normal:
      - keep points within z_tol of median z
      - fit plane via SVD
    Returns unit normal n (3,).
    """
    if len(pos3d) < 3:
        return np.array([0.0, 0.0, 1.0])

    z = pos3d[:, 2]
    z0 = np.median(z)
    mask = np.abs(z - z0) < z_tol
    pts = pos3d[mask]
    if len(pts) < 3:
        pts = pos3d

    origin = pts.mean(axis=0)
    X = pts - origin
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    n = VT[-1]
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / nn


def project_vector_to_plane(vec3d, n_unit):
    """
    v_parallel = v - (v·n) n
    """
    return vec3d - np.dot(vec3d, n_unit) * n_unit


# ================================
# WS REFERENCE AREAS
# ================================
def compute_ws_reference_areas(atoms_ref):
    tags = np.array(atoms_ref.get_tags())
    surface_atoms = atoms_ref[tags == SURFACE_TAG]
    if len(surface_atoms) == 0:
        return {}

    # best-fit plane normal for *reference* slab (used to convert A2D -> A_phys)
    n_ref = median_filtered_plane_normal(surface_atoms.positions, Z_TOL_PLANE)
    nz_ref = max(abs(n_ref[2]), 1e-8)

    # window size estimate kept as in your code (uses XY components of MIC vector)
    d_est = np.median([
        np.linalg.norm(mic_vec(surface_atoms, 0, j)[:2])
        for j in range(1, min(7, len(surface_atoms)))
    ])
    L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE

    i_center = find_central_surface_atom(surface_atoms)
    rep_pos3d, _, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)

    cell = surface_atoms.get_cell()
    b1, b2 = lattice_xy_basis_from_cell(cell)

    rep_pos2d = project_positions_lattice(rep_pos3d, b1, b2)
    base_pos2d = project_positions_lattice(surface_atoms.positions, b1, b2)

    shift_central2d = project_positions_lattice(
        (((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]).reshape(1, 3)),
        b1, b2
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

        # same inside-window logic as before
        if SPoly(poly - np.array([x0, y0])).within(window):
            A2d = polygon_area(poly)
            Aphys = A2d / nz_ref  # physical area on the best-fit plane
            area_ref[idx % n_surf] = Aphys

    return area_ref


# ================================
# BRIDGE BONDS
# ================================
def build_bridge_bonds(surface_atoms, n_plane):
    """
    Bond length is computed as the length of the bond vector projected onto the
    best-fit surface plane (using normal n_plane).
    Plotting direction (vec_ab) stays unchanged (still based on MIC vector).
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

            v_par = project_vector_to_plane(vec, n_plane)
            d = np.linalg.norm(v_par)

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
    #rows = rows = [r for r in db.select() if (r.id == 95 or r.id > 4317)]

    # --- select pure Cu and pure Au reference slabs ---
    atoms_ref_cu = next(
        r.toatoms()
        for r in rows
        if "Cu" in r.toatoms().get_chemical_symbols()
        and "Au" not in r.toatoms().get_chemical_symbols()
    )
    atoms_ref_au = next(
        r.toatoms()
        for r in rows
        if "Au" in r.toatoms().get_chemical_symbols()
        and "Cu" not in r.toatoms().get_chemical_symbols()
    )

    # reference areas now stored as physical areas (A2D/|n_z|) on the reference slab plane
    area_ref_cu = compute_ws_reference_areas(atoms_ref_cu)
    area_ref_au = compute_ws_reference_areas(atoms_ref_au)

    for row in rows:
        atoms = row.toatoms()
        au_surf, au_sub = au_composition(atoms, SURFACE_TAG)
        tags = np.array(atoms.get_tags())
        surface_atoms = atoms[tags == SURFACE_TAG]
        if len(surface_atoms) == 0:
            continue

        # --- geometry ---
        d_est = np.median([
            np.linalg.norm(mic_vec(surface_atoms, 0, j)[:2])
            for j in range(1, min(7, len(surface_atoms)))
        ])
        L = PLOT_L_OVERRIDE or (PLOT_N_ATOMS - 1) * d_est * PLOT_SCALE
        i_center = find_central_surface_atom(surface_atoms)

        rep_pos3d, rep_sym, tile_id = repeat_surface(surface_atoms, MAX_REPEAT, MAX_REPEAT)

        cell = surface_atoms.get_cell()
        b1, b2 = lattice_xy_basis_from_cell(cell)

        rep_pos2d = project_positions_lattice(rep_pos3d, b1, b2)
        base_pos2d = project_positions_lattice(surface_atoms.positions, b1, b2)

        shift_central2d = project_positions_lattice(
            (((MAX_REPEAT // 2) * cell[0] + (MAX_REPEAT // 2) * cell[1]).reshape(1, 3)),
            b1, b2
        )[0]

        center2d = base_pos2d[i_center] + shift_central2d
        x0, y0 = center2d - 0.5 * L
        dx = 0.5 * (L - PLOT_L_VIEW)
        x0v, y0v = x0 + dx, y0 + dx
        window_view = box(0, 0, PLOT_L_VIEW, PLOT_L_VIEW)

        # lattice-based tile estimate (plot-safe, no PCA)
        a1_xy = project_positions_lattice(cell[0].reshape(1, 3), b1, b2)[0]
        n_tile = int(np.ceil(PLOT_L_VIEW / np.linalg.norm(a1_xy))) + 3

        # --- plane normal for this slab (used for physical WS area + bond strain)
        n_plane = median_filtered_plane_normal(surface_atoms.positions, Z_TOL_PLANE)
        nz_plane = max(abs(n_plane[2]), 1e-8)

        # --- Voronoi ---
        vor = Voronoi(rep_pos2d)
        central_tile = np.array([MAX_REPEAT // 2, MAX_REPEAT // 2])
        central_mask = np.all(tile_id == central_tile, axis=1)

        central_polys, poly_vals = [], []
        surface_syms = surface_atoms.get_chemical_symbols()
        n_surf = len(surface_atoms)

        for idx in np.where(central_mask)[0]:
            reg = vor.regions[vor.point_region[idx]]
            if -1 in reg or len(reg) < 3:
                continue

            poly = order_polygon_ccw(vor.vertices[reg])
            local = idx % n_surf

            # choose the same reference area dictionary as before, but now Aref is physical
            if surface_syms[local] == "Au":
                if local not in area_ref_au:
                    continue
                Aref = area_ref_au[local]
            else:
                if local not in area_ref_cu:
                    continue
                Aref = area_ref_cu[local]

            # --- CHANGED: physical area on the best-fit plane ---
            A2d = polygon_area(poly)
            A = A2d / nz_plane

            central_polys.append(poly)
            poly_vals.append(100.0 * (A - Aref) / Aref)

        poly_vals = np.array(poly_vals)

        # --- Bonds (plane-projected length for strain, plotting unchanged) ---
        bonds = build_bridge_bonds(surface_atoms, n_plane)
        bond_lines, bond_vals = [], []

        for ix in range(-n_tile, n_tile + 1):
            for iy in range(-n_tile, n_tile + 1):
                shift = project_positions_lattice(
                    ((ix * cell[0] + iy * cell[1]).reshape(1, 3)),
                    b1, b2
                )[0]
                for b in bonds:
                    p0 = base_pos2d[b["i"]] + shift_central2d + shift - np.array([x0v, y0v])
                    p1 = p0 + project_vector_lattice(b["vec_ab"], b1, b2)
                    if ((0 <= p0[0] <= PLOT_L_VIEW and 0 <= p0[1] <= PLOT_L_VIEW) or (0 <= p1[0] <= PLOT_L_VIEW and 0 <= p1[1] <= PLOT_L_VIEW)):
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

        # --- plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        for poly, val in zip(central_polys, poly_vals):
            for ix in range(-n_tile, n_tile + 1):
                for iy in range(-n_tile, n_tile + 1):
                    shift = project_positions_lattice(
                        ((ix * cell[0] + iy * cell[1]).reshape(1, 3)),
                        b1, b2
                    )[0]
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

        # --- optimized bonds (NO alpha) ---
        if bond_lines:
            ax.add_collection(LineCollection(
                bond_lines,
                colors=BOND_OUTLINE_COLOR,
                linewidths=BOND_WIDTH_OUTLINE,
                zorder=3,
                capstyle="round",
                joinstyle="round"
            ))

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
        ax.set_title(
            f"Surface Au {au_surf:.0f}% | Subsurface Au {au_sub:.0f}%",
            fontsize=12
        )
        ax.set_aspect("equal")

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

        minor_ticks = [-3.75, -1.25, 1.25, 3.75]
        cbar.ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

        cbar.ax.tick_params(which="major", length=8, width=1.3)
        cbar.ax.tick_params(which="minor", length=4, width=0.8)

        cbar.set_label("Strain (%)", rotation=90, labelpad=12)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"strain_row_{row.id}.svg"), dpi=200)
        plt.close()
        print("cell a1,a2 z-components:", cell[0][2], cell[1][2])
        print("b1·b1,b2·b2,b1·b2:", np.dot(b1,b1), np.dot(b2,b2), np.dot(b1,b2))
        print("n_plane:", n_plane, "|n_z|:", abs(n_plane[2]))
        print("central points:", np.sum(central_mask), "polys plotted:", len(central_polys), "bonds plotted:", len(bond_lines))
        print("unique bonds in network:", len(bonds))
        print("ref coverage Au:", len(area_ref_au), "Cu:", len(area_ref_cu))



print("All strain plots saved to:", OUTDIR)

