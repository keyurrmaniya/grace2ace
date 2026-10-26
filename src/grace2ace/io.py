"""File helpers shared by all stages.

Stages hand structures to each other as extxyz, never as pickles: the GRACE
env (numpy 2) and the pacemaker env (numpy 1.26) cannot read each other's
pickles. Only ``build_df`` writes pickles, and it runs in the pacemaker env.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

# info keys kept when writing extxyz. ASSYST also stores 'lineage' (a list)
# and pyxtal objects, which extxyz cannot round-trip.
INFO_KEYS = {
    "uuid": "uuid",
    "seed": "family",          # ASSYST: uuid of the root pyxtal structure
    "family": "family",
    "parent": "parent",
    "step": "step",
    "perturbation": "perturbation",
    "spacegroup": "spacegroup",
    "requested spacegroup": "requested_spacegroup",
    "requested_spacegroup": "requested_spacegroup",
    "config_type": "config_type",
    "relax_converged": "relax_converged",
    "origin": "origin",        # seed:<name> / boost (see assyst.seeds / assyst.boost)
    "split": "split",          # train/test side fixed by a reused parent (boost copies)
    "relax_epa": "relax_epa",  # teacher energy per atom at the end of the relaxation
}


def subsystem(atoms) -> str:
    """'Al-Ni-W'-style label of the elements present."""
    return "-".join(sorted(set(atoms.get_chemical_symbols())))


def clean_info(atoms):
    """Keep only scalar, extxyz-safe info entries under stable names."""
    info = {}
    for key, value in atoms.info.items():
        new = INFO_KEYS.get(key)
        if new is None:
            continue
        # ASSYST sets 'seed' to a fresh uuid on any structure without one,
        # e.g. a perturbed copy read back from extxyz; the stored family wins.
        if key == "seed" and "family" in atoms.info:
            continue
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (str, int, float, bool)):
            info[new] = value
    atoms.info.clear()
    atoms.info.update(info)
    return atoms


def write_extxyz(path, structures: Iterable):
    import ase.io

    structures = list(structures)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if structures:
        ase.io.write(tmp, structures, format="extxyz")
    else:
        tmp.write_text("")
    os.replace(tmp, path)
    return len(structures)


def read_extxyz(path):
    import ase.io

    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return []
    return ase.io.read(path, index=":", format="extxyz")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ---- cell shape ---------------------------------------------------------------

# pacemaker's C++ neighbour list (pyace ace_atoms_binding.cpp, inv3x3) inverts the cell and
# rejects it if |A.A^-1 - I| > 1e-14 in ANY element. Rounding error grows with the condition
# number, so a valid but needle-shaped cell (e.g. angles 173/3/171 deg, cond ~85, as a relaxation
# can leave behind) fails. A reduced cell describes the same crystal with short, near-orthogonal
# vectors (cond ~2).
MAX_CELL_CONDITION = 10.0


def pacemaker_cell_ok(cell) -> bool:
    """The exact inv3x3 test pacemaker applies (same formula, same 1e-14 limit)."""
    a = [[float(x) for x in row] for row in np.asarray(cell)]
    det = (a[0][0] * (a[2][2] * a[1][1] - a[2][1] * a[1][2])
           - a[1][0] * (a[2][2] * a[0][1] - a[2][1] * a[0][2])
           + a[2][0] * (a[1][2] * a[0][1] - a[1][1] * a[0][2]))
    if abs(det) < 1e-9:
        return False
    inv = [[(a[2][2] * a[1][1] - a[2][1] * a[1][2]) / det, -(a[2][2] * a[0][1] - a[2][1] * a[0][2]) / det,
            (a[1][2] * a[0][1] - a[1][1] * a[0][2]) / det],
           [-(a[2][2] * a[1][0] - a[2][0] * a[1][2]) / det, (a[2][2] * a[0][0] - a[2][0] * a[0][2]) / det,
            -(a[1][2] * a[0][0] - a[1][0] * a[0][2]) / det],
           [(a[2][1] * a[1][0] - a[2][0] * a[1][1]) / det, -(a[2][1] * a[0][0] - a[2][0] * a[0][1]) / det,
            (a[1][1] * a[0][0] - a[1][0] * a[0][1]) / det]]
    return all(abs(sum(a[i][k] * inv[k][j] for k in range(3)) - (1.0 if i == j else 0.0)) <= 1e-14
               for i in range(3) for j in range(3))


def reduce_cell(atoms) -> bool:
    """Minkowski-reduce a badly shaped periodic cell in place; True if it was changed.

    Only the cell vectors change (by an integer matrix) and atoms are wrapped back; the
    Cartesian frame is kept, so energy, forces and stress labels stay valid. (ase's
    niggli_reduce would also ROTATE the cell and so invalidate Cartesian forces.)
    """
    from ase.geometry import minkowski_reduce

    if not atoms.pbc.all():
        return False
    cell = atoms.cell[:]
    if np.linalg.cond(cell) <= MAX_CELL_CONDITION and pacemaker_cell_ok(cell):
        return False
    rcell, _ = minkowski_reduce(cell)
    atoms.set_cell(rcell, scale_atoms=False)
    atoms.wrap()
    return True


# ---- run-directory layout -------------------------------------------------

def stage_dir(run_dir, stage) -> Path:
    return Path(run_dir) / stage


def shard_name(shard: int) -> str:
    return f"shard_{shard:03d}"


def done_marker(run_dir, stage, shard=None) -> Path:
    d = stage_dir(run_dir, stage)
    return d / (f"{shard_name(shard)}.done" if shard is not None else "DONE")


def is_done(run_dir, stage, shard=None) -> bool:
    return done_marker(run_dir, stage, shard).exists()


def mark_done(run_dir, stage, shard=None, info=None):
    marker = done_marker(run_dir, stage, shard)
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, info or {})


def shards_done(run_dir, stage, n_shards) -> list[int]:
    return [i for i in range(n_shards) if is_done(run_dir, stage, i)]


def state_path(run_dir) -> Path:
    return Path(run_dir) / "state.json"


def load_state(run_dir) -> dict:
    return read_json(state_path(run_dir), default={"jobs": [], "fit_attempt": 0})


def save_state(run_dir, state):
    write_json(state_path(run_dir), state)


# ---- where a run's dataset lives (its own build_df/, or dataset.reuse_from's) ----

def generates_structures(cfg) -> bool:
    """Does this run make and label new structures (generate + label stages)?

    Without reuse_from: random crystals and/or seeds. With reuse_from: only seeds and boost
    copies (old frozen configs have neither key, so a plain reuse stays a plain reuse).
    """
    a = cfg.get("assyst") or {}
    if (cfg.get("dataset") or {}).get("reuse_from"):
        return bool(a.get("seeds")) or bool(a.get("boost"))
    return bool(a.get("random_crystals", True)) or bool(a.get("seeds"))


def reuse_copy(cfg) -> bool:
    """reuse_from, but this run builds its OWN dataset: the reused one filtered
    (reuse_filters) and/or plus newly generated structures (seeds, boost)."""
    ds = cfg.get("dataset") or {}
    return bool(ds.get("reuse_from")) and (bool(ds.get("reuse_filters")) or generates_structures(cfg))


reuse_filtered = reuse_copy     # older name


def dataset_source(cfg, run_dir) -> Path:
    """The run directory whose build_df/ holds this run's dataset."""
    src = (cfg.get("dataset") or {}).get("reuse_from")
    return Path(src) if src and not reuse_copy(cfg) else Path(run_dir)


def reuse_source_ready(cfg) -> bool:
    """Is the dataset named by dataset.reuse_from finished (True when nothing is reused)?"""
    src = (cfg.get("dataset") or {}).get("reuse_from")
    return True if not src else is_done(src, "build_df")


def data_dir(cfg, run_dir) -> Path:
    return stage_dir(dataset_source(cfg, run_dir), "build_df")


def dataset_ready(cfg, run_dir) -> bool:
    return is_done(dataset_source(cfg, run_dir), "build_df")
