import numpy as np
import matplotlib.pyplot as plt

from ase.db import connect
from ase.geometry import find_mic
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

# ================================
# USER SETTINGS
# ================================
DB_PATH = "test.db"
SURFACE_TAG = 1

N_NEIGHBORS = 6
LINEWIDTH = 3.0
ATOM_SIZE = 137.0

# real-space square patch size (Å)
SQUARE_SIZE = 15.0

# initial repeats (auto-increased if needed)
REPEAT_X0 = 2
REPEAT_Y0 = 2
MAX_REPEAT = 6          # <-- raise this a bit so centering has room
COVERAGE_MARGIN = 4.0   # Å

ATOM_COLORS = {"Au": "orange", "Cu": "brown"}

D_REF = {
    ("Au", "Au"): 2.980,
    ("Au", "Cu"): 2.79450000000000000000,
    ("Cu", "Cu"): 2.609
}

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
    """Return repeated positions/symbols for plotting (no need to keep indices)."""
    pos = surface_atoms.positions
    cell = surface_atoms.get_cell()
    symbols = surface_atoms.get_chemical_symbols()

    all_pos = []
    all_sym = []

    for ix in range(nx):
        for iy in range(ny):
            shift = ix * cell[0] + iy * cell[1]
            all_pos.append(pos + shift)
            all_sym.extend(symbols)

    return np.vstack(all_pos), all_sym


def surface_basis_from_positions(pos):
    xy = pos[:, :2] - pos[:, :2].mean(axis=0)
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e1 = eigvecs[:, order[0]]
    e2 = eigvecs[:, order[1]]
    return e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)


def project_positions(pos, e1, e2):
    xy = pos[:, :2]
    return np.column_stack((xy @ e1, xy @ e2))


def project_vector(vec, e1, e2):
    v = vec[:2]
    return np.array([np.dot(v, e1), np.dot(v, e2)])


def find_central_surface_atom(surface_atoms):
    """
    Central surface atom = the surface atom whose XY is closest to the XY centroid
    of all surface atoms (robust for general surface shapes).
    Returns: i_center_surf (index in surface_atoms)
    """
    xy = surface_atoms.positions[:, :2]
    xy0 = xy.mean(axis=0)
    i_center = int(np.argmin(np.sum((xy - xy0) ** 2, axis=1)))
    return i_center


# ================================
# BOND CONSTRUCTION
# ================================
def build_bridge_bonds(surface_atoms):
    symbols = surface_atoms.get_chemical_symbols()
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

            pair = tuple(sorted((symbols[i], symbols[j])))
            if pair not in D_REF:
                continue

            pct = 100.0 * (d - D_REF[pair]) / D_REF[pair]

            bond_dict[(a, b)] = {
                "i": a,
                "vec_ab": vec if a == i else -vec,
                "pct_dev": pct
            }

    return list(bond_dict.values())


# ================================
# GLOBAL STRAIN RANGE (DB-WIDE)
# ================================
def compute_global_strain_range(db_path):
    all_strains = []

    with connect(db_path) as db:
        for row in db.select():
            atoms = row.toatoms()
            tags = np.array(atoms.get_tags())
            surface_atoms = atoms[tags == SURFACE_TAG]
            if len(surface_atoms) == 0:
                continue

            bonds = build_bridge_bonds(surface_atoms)
            for b in bonds:
                if np.isfinite(b["pct_dev"]):
                    all_strains.append(b["pct_dev"])

    if len(all_strains) == 0:
        raise RuntimeError("No strain data found in database")

    all_strains = np.array(all_strains)
    p_lo, p_hi = np.percentile(all_strains, [2, 98])
    vmax = max(abs(p_lo), abs(p_hi))

    print(
        f"Global strain range (2–98 percentile): "
        f"{-vmax:.2f}% → {vmax:.2f}%"
    )

    return vmax


GLOBAL_VMAX = compute_global_strain_range(DB_PATH)
GLOBAL_NORM = TwoSlopeNorm(
    vmin=-GLOBAL_VMAX,
    vcenter=0.0,
    vmax= GLOBAL_VMAX
)

# ================================
# DOMAIN SELECTION (CENTERED PATCH)
# ================================
def ensure_covering_repeat_and_centered_window(surface_atoms, L, i_center_surf):
    """
    Choose nx,ny big enough that the LxL window centered on the chosen atom
    is fully covered by the repeated domain (with margin).
    Returns nx, ny, e1, e2, rep_pos2d, rep_symbols, x0, y0, center2d
    """
    nx, ny = REPEAT_X0, REPEAT_Y0

    while True:
        rep_pos, rep_symbols = repeat_surface(surface_atoms, nx, ny)

        # basis (use repeated positions for stability)
        e1, e2 = surface_basis_from_positions(rep_pos)

        rep_pos2d = project_positions(rep_pos, e1, e2)

        # projected base positions (single cell)
        base_pos2d = project_positions(surface_atoms.positions, e1, e2)

        # pick a reference tile near the middle of the repeated patch
        cell = surface_atoms.get_cell()
        ix_ref = nx // 2
        iy_ref = ny // 2
        shift3d_ref = ix_ref * cell[0] + iy_ref * cell[1]
        shift2d_ref = project_positions(shift3d_ref.reshape(1, 3), e1, e2)[0]

        center2d = base_pos2d[i_center_surf] + shift2d_ref

        x0 = center2d[0] - 0.5 * L
        y0 = center2d[1] - 0.5 * L

        xmin, ymin = rep_pos2d.min(axis=0)
        xmax, ymax = rep_pos2d.max(axis=0)

        # require that [x0, x0+L] and [y0, y0+L] fit well inside repeated bounds
        ok_x = (x0 >= xmin + COVERAGE_MARGIN) and (x0 + L <= xmax - COVERAGE_MARGIN)
        ok_y = (y0 >= ymin + COVERAGE_MARGIN) and (y0 + L <= ymax - COVERAGE_MARGIN)

        if ok_x and ok_y:
            return nx, ny, e1, e2, rep_pos2d, rep_symbols, x0, y0, center2d

        if nx >= MAX_REPEAT and ny >= MAX_REPEAT:
            # fall back: still return, but window may be truncated at edges
            return nx, ny, e1, e2, rep_pos2d, rep_symbols, x0, y0, center2d

        # grow repeats where needed
        if not ok_x:
            nx += 1
        if not ok_y:
            ny += 1


def clip_atoms(pos2d, symbols, x0, y0, L):
    mask = (
        (pos2d[:, 0] >= x0) & (pos2d[:, 0] <= x0 + L) &
        (pos2d[:, 1] >= y0) & (pos2d[:, 1] <= y0 + L)
    )
    return pos2d[mask], [s for s, m in zip(symbols, mask) if m]


def clip_lines(lines, colors, x0, y0, L):
    new_lines, new_colors = [], []
    x1, y1 = x0 + L, y0 + L

    for (p0, p1), c in zip(lines, colors):
        if (
            x0 <= p0[0] <= x1 and y0 <= p0[1] <= y1 and
            x0 <= p1[0] <= x1 and y0 <= p1[1] <= y1
        ):
            new_lines.append([p0, p1])
            new_colors.append(c)

    return new_lines, new_colors


# ================================
# MAIN LOOP (PLOTTING)
# ================================
with connect(DB_PATH) as db:
    for row in db.select():
        atoms = row.toatoms()
        tags = np.array(atoms.get_tags())
        surf_indices = np.where(tags == SURFACE_TAG)[0]
        surface_atoms = atoms[surf_indices]
        if len(surface_atoms) == 0:
            continue

        # --- find & verify central surface atom index ---
        i_center_surf = find_central_surface_atom(surface_atoms)
        i_center_global = int(surf_indices[i_center_surf])

        print(
            f"[row {row.id}] central surface atom: "
            f"surf_index={i_center_surf}, global_index={i_center_global}, "
            f"matches_87={i_center_global == 87}"
        )

        bonds = build_bridge_bonds(surface_atoms)

        nx, ny, e1, e2, rep_pos2d, rep_symbols, x0, y0, center2d = (
            ensure_covering_repeat_and_centered_window(surface_atoms, SQUARE_SIZE, i_center_surf)
        )

        base_pos2d = project_positions(surface_atoms.positions, e1, e2)
        cell = surface_atoms.get_cell()

        lines, colors = [], []
        for ix in range(nx):
            for iy in range(ny):
                shift3d = ix * cell[0] + iy * cell[1]
                shift2d = project_positions(shift3d.reshape(1, 3), e1, e2)[0]

                for b in bonds:
                    p0 = base_pos2d[b["i"]] + shift2d
                    dv = project_vector(b["vec_ab"], e1, e2)
                    p1 = p0 + dv
                    lines.append([p0, p1])
                    colors.append(b["pct_dev"])

        clipped_pos2d, clipped_symbols = clip_atoms(
            rep_pos2d, rep_symbols, x0, y0, SQUARE_SIZE
        )
        lines, colors = clip_lines(lines, colors, x0, y0, SQUARE_SIZE)

        # translate patch to (0, 0)
        clipped_pos2d -= np.array([x0, y0])
        lines = [[p0 - np.array([x0, y0]), p1 - np.array([x0, y0])]
                 for (p0, p1) in lines]

        # for sanity: the chosen center should now be at (L/2, L/2) in this coordinate system
        center_in_patch = (center2d - np.array([x0, y0]))

        fig, ax = plt.subplots(figsize=(6, 6))

        lc = LineCollection(
            lines,
            cmap="seismic",
            linewidths=LINEWIDTH,
            norm=GLOBAL_NORM
        )
        lc.set_array(np.array(colors))
        ax.add_collection(lc)

        ax.scatter(
            clipped_pos2d[:, 0],
            clipped_pos2d[:, 1],
            c=[ATOM_COLORS.get(s, "gray") for s in clipped_symbols],
            s=ATOM_SIZE,
            edgecolors="black",
            zorder=3
        )

        # optional: mark the patch center (should coincide with central surface atom)
        ax.scatter(
            [center_in_patch[0]],
            [center_in_patch[1]],
            s=ATOM_SIZE * 3.0,
            facecolors="none",
            edgecolors="black",
            linewidths=2.0,
            zorder=4
        )

        ax.set_xlim(0, SQUARE_SIZE)
        ax.set_ylim(0, SQUARE_SIZE)
        ax.set_aspect("equal")
        ax.set_title(
            f"Row {row.id} – surface strain "
            f"({SQUARE_SIZE}×{SQUARE_SIZE} Å patch)\n"
            f"centered on global surface atom {i_center_global}"
        )
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")

        plt.colorbar(lc, ax=ax, label="Δd / d₀ (%)")
        plt.tight_layout()
        plt.show()

print("Done.")

