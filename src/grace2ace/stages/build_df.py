"""Stage 3 (pacemaker env): labelled extxyz -> pacemaker train/test DataFrames.

* ``energy_corrected`` uses a FROZEN reference energy (same least-squares +
  shift as pacemaker's ``reference_energy: auto``, computed once), so energies
  of different runs and later refits stay comparable.
* The test split is by structure FAMILY (all structures grown from the same
  random crystal), so no test structure has a near-twin in training.
* ``hull_distance`` (eV/atom above the convex hull, as pacemaker computes it)
  is stored for the nfit check and for hull-binned errors.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from .. import io

log = logging.getLogger(__name__)
PICKLE_PROTOCOL = 4


def _rows_from_extxyz(structures, source):
    rows = []
    for at in structures:
        calc = at.calc
        if calc is None or "energy" not in calc.results or "forces" not in calc.results:
            raise ValueError(f"{source}: structure without energy/forces")
        e = float(calc.results["energy"])
        f = np.array(calc.results["forces"])
        stress = calc.results.get("stress")
        info = dict(at.info)
        at = at.copy()
        at.calc = None
        rows.append({
            "ase_atoms": at,
            "energy": e,
            "forces": f,
            "stress": None if stress is None else np.array(stress),
            "name": f"{source}/{info.get('config_type', 'unknown')}/{info.get('uuid', len(rows))}",
            "config_type": info.get("config_type", "unknown"),
            "family": info.get("family") or info.get("uuid") or f"{source}:{len(rows)}",
            "origin": info.get("origin", "random"),
            "split": info.get("split"),
        })
    return rows


def _rows_from_extra(extra):
    if extra["format"] == "extxyz":
        rows = _rows_from_extxyz(io.read_extxyz(extra["path"]), extra["name"])
        for r in rows:
            r["config_type"] = extra["name"]
        return rows
    df = pd.read_pickle(extra["path"], compression="infer")
    for col in ("ase_atoms", "energy", "forces"):
        if col not in df.columns:
            raise ValueError(f"extra_data {extra['path']}: column '{col}' missing")
    group = extra.get("group_column")
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        rows.append({
            "ase_atoms": r["ase_atoms"],
            "energy": float(r["energy"]),
            "forces": np.array(r["forces"]),
            "stress": None,
            "name": f"{extra['name']}/{i}",
            "config_type": extra["name"],
            "family": f"{extra['name']}:{r[group] if group else i}",
        })
    return rows


def composition_matrix(atoms_list, elements):
    index = {e: i for i, e in enumerate(elements)}
    comp = np.zeros((len(atoms_list), len(elements)))
    for row, at in enumerate(atoms_list):
        for sym, n in Counter(at.get_chemical_symbols()).items():
            comp[row, index[sym]] = n
    return comp


def freeze_reference(df, elements) -> dict:
    """pacemaker's ``reference_energy: auto``, done once (pyace process_df.py).

    Per-element energies by least squares, then a shift that puts the
    largest-volume structure at zero corrected energy.
    """
    comp = composition_matrix(df["ase_atoms"], elements)
    e = df["energy"].to_numpy()
    esa = np.linalg.pinv(comp, rcond=1e-10) @ e
    corrected_pa = (e - comp @ esa) / df["natoms"].to_numpy()
    vpa = df["ase_atoms"].map(lambda a: a.get_volume() / len(a)).to_numpy()
    shift = -float(corrected_pa[int(np.argmax(vpa))])
    ref = {el: float(v) for el, v in zip(elements, esa)}
    ref["shift"] = shift
    return ref


def apply_reference(df, elements, ref):
    comp = composition_matrix(df["ase_atoms"], elements)
    esa = np.array([ref.get(el, 0.0) for el in elements])
    df["energy_corrected"] = df["energy"].to_numpy() - comp @ esa + ref.get("shift", 0.0) * df["natoms"].to_numpy()
    return df


def hull_distance(df):
    """Distance above the convex hull with pacemaker's own routine (None if pyace is missing)."""
    try:
        from pyace.process_df import compute_convexhull_dist
    except ImportError:
        log.warning("pyace not importable: no hull distances (nfit check and hull bins skipped)")
        return None
    tmp = pd.DataFrame({"ase_atoms": df["ase_atoms"],
                        "epa": df["energy_corrected"] / df["natoms"]})
    compute_convexhull_dist(tmp, energy_per_atom_column="epa", verbose=False)
    return tmp["e_chull_dist_per_atom"].to_numpy()


def family_split(df, fraction, seed):
    """Whole families go to test until ``fraction`` of the structures is reached."""
    rng = np.random.default_rng(seed)
    sizes = df.groupby("family").size()
    families = sizes.index.to_numpy()[rng.permutation(len(sizes))]
    target = fraction * len(df)
    test, count = set(), 0
    for fam in families:
        if count >= target:
            break
        test.add(fam)
        count += sizes[fam]
    return df["family"].isin(test).to_numpy()


def filter_mask(df, limits: dict) -> tuple[np.ndarray, dict]:
    """Rows inside ``limits`` (keys as in the `filters` config) and why the others fail."""
    vpa = df["ase_atoms"].map(lambda a: a.get_volume() / len(a)).to_numpy()
    epa = (df["energy"] / df["natoms"]).to_numpy()
    tests = {
        "max_volume_per_atom": lambda x: vpa <= x,
        "min_energy_per_atom": lambda x: epa >= x,
        "max_energy_per_atom": lambda x: epa <= x,
        "max_force": lambda x: df["fmax"].to_numpy() <= x,
    }
    keep = np.ones(len(df), dtype=bool)
    reasons = {}
    for key, limit in limits.items():
        ok = tests[key](limit)
        reasons[key] = int((~ok).sum())
        keep &= ok
    return keep, reasons


def _new_rows(cfg, run_dir):
    """Structures this run generated and labelled (seeds, boost copies) as a DataFrame."""
    n_shards = int(cfg["assyst"]["n_shards"])
    missing = sorted(set(range(n_shards)) - set(io.shards_done(run_dir, "label", n_shards)))
    if missing:
        raise RuntimeError(f"label shards not done: {missing}")
    rows = []
    for i in range(n_shards):
        rows += _rows_from_extxyz(io.read_extxyz(io.stage_dir(run_dir, "label") / f"{io.shard_name(i)}.extxyz"),
                                  "assyst")
    df = pd.DataFrame(rows)
    if len(df):
        df["natoms"] = df["ase_atoms"].map(len)
        df["subsystem"] = df["ase_atoms"].map(io.subsystem)
        df["fmax"] = df["forces"].map(lambda f: float(np.linalg.norm(f, axis=1).max()))
    return df


def _run_reuse_copy(cfg: dict, run_dir):
    """This run's own dataset from dataset.reuse_from's: filtered (reuse_filters) and/or
    plus the structures this run generated (seeds, boost copies). Same reference energy;
    reused structures keep their train/test side, boost copies follow their parent,
    new seed families are split by family."""
    src_run = cfg["dataset"]["reuse_from"]
    if not io.is_done(src_run, "build_df"):
        raise RuntimeError(f"dataset.reuse_from {src_run}/build_df is not done")
    src = io.stage_dir(src_run, "build_df")
    limits = cfg["dataset"].get("reuse_filters") or {}
    train = pd.read_pickle(src / "train.pckl.gzip", compression="gzip")
    test = pd.read_pickle(src / "test.pckl.gzip", compression="gzip")
    new = _new_rows(cfg, run_dir) if io.generates_structures(cfg) else pd.DataFrame()
    elements = list(cfg["system"]["elements"])
    parts = [train, test] + ([new] if len(new) else [])
    stray = set().union(*pd.concat(parts)["ase_atoms"].map(lambda a: set(a.get_chemical_symbols()))) - set(elements)
    if stray:
        raise ValueError(f"structures contain elements not in system.elements: {sorted(stray)}")

    report = {"reused_from": str(src_run), "limits": limits}
    kept = []
    for split, df in (("train", train), ("test", test), ("new", new)):
        if not len(df):
            continue
        keep, reasons = filter_mask(df, limits)
        report[split] = {
            "before": len(df), "after": int(keep.sum()), "removed_by": reasons,
            "removed_by_subsystem": df[~keep].groupby("subsystem").size().to_dict(),
            "removed_by_config_type": df[~keep].groupby("config_type").size().to_dict(),
        }
        log.info("%s: %d -> %d structures (%s)", split, len(df), keep.sum(), reasons)
        df = df[keep].copy()
        if split == "new":
            fixed = df["split"].isin(["train", "test"])
            is_test = (df["split"] == "test").to_numpy()
            free = ~fixed.to_numpy()
            if free.any():
                is_test[free] = family_split(df[free], float(cfg["dataset"]["test_fraction"]), cfg["system"]["seed"])
            df["_test"] = is_test
            report["new"]["by_origin"] = df["origin"].map(lambda o: o.split(":")[0]).value_counts().to_dict()
            report["new"]["test"] = int(is_test.sum())
        else:
            df["_test"] = split == "test"
        kept.append(df)
    df = pd.concat(kept).reset_index(drop=True)
    if not len(df):
        raise RuntimeError("no structures left to fit")

    # same frozen reference energy as the source, so energies stay comparable;
    # the convex hull is recomputed because the structure set changed
    ref = io.read_json(src / "reference_energy.json")
    apply_reference(df, elements, ref)
    hd = hull_distance(df)
    if hd is not None:
        df["hull_distance"] = hd
    is_test = df.pop("_test").to_numpy().astype(bool)
    _write_dataset(cfg, run_dir, df, is_test, ref, extra_summary={"reuse": report})
    io.write_json(io.stage_dir(run_dir, "build_df") / "reuse_filter_report.json", report)


def run(cfg: dict, run_dir, shard=None):
    if io.is_done(run_dir, "build_df"):
        log.info("build_df already done")
        return
    if io.reuse_copy(cfg):
        return _run_reuse_copy(cfg, run_dir)
    n_shards = int(cfg["assyst"]["n_shards"])
    missing = sorted(set(range(n_shards)) - set(io.shards_done(run_dir, "label", n_shards)))
    if missing:
        raise RuntimeError(f"label shards not done: {missing}")

    rows = []
    for i in range(n_shards):
        path = io.stage_dir(run_dir, "label") / f"{io.shard_name(i)}.extxyz"
        rows += _rows_from_extxyz(io.read_extxyz(path), "assyst")
    n_assyst = len(rows)
    for extra in cfg["extra_data"]:
        rows += _rows_from_extra(extra)
    log.info("%d ASSYST structures + %d extra", n_assyst, len(rows) - n_assyst)
    if not rows:
        raise RuntimeError("no structures to fit")

    df = pd.DataFrame(rows)
    df["natoms"] = df["ase_atoms"].map(len)
    df["subsystem"] = df["ase_atoms"].map(io.subsystem)
    df["fmax"] = df["forces"].map(lambda f: float(np.linalg.norm(f, axis=1).max()))

    elements = list(cfg["system"]["elements"])
    stray = set().union(*df["ase_atoms"].map(lambda a: set(a.get_chemical_symbols()))) - set(elements)
    if stray:
        raise ValueError(f"structures contain elements not in system.elements: {sorted(stray)}")

    ref = cfg["dataset"]["reference_energy"]
    ref = freeze_reference(df, elements) if ref == "freeze" else {k: float(v) for k, v in ref.items()}
    apply_reference(df, elements, ref)
    hd = hull_distance(df)
    if hd is not None:
        df["hull_distance"] = hd

    is_test = family_split(df, float(cfg["dataset"]["test_fraction"]), cfg["system"]["seed"])
    _write_dataset(cfg, run_dir, df, is_test, ref)


def _write_dataset(cfg, run_dir, df, is_test, ref, extra_summary=None):
    """train/test pickles, reference energy, energetics table, summary, plot, DONE."""
    out = io.stage_dir(run_dir, "build_df")
    out.mkdir(parents=True, exist_ok=True)
    # needle-shaped cells crash pacemaker's neighbour list (see io.reduce_cell); same crystal,
    # same labels, better cell
    reduced = int(df["ase_atoms"].map(io.reduce_cell).sum())
    if reduced:
        log.info("reduced %d badly shaped cells (condition number > %g)", reduced, io.MAX_CELL_CONDITION)
    bad = int((~df["ase_atoms"].map(lambda a: io.pacemaker_cell_ok(a.cell[:]))).sum())
    if bad:
        raise RuntimeError(f"{bad} cells still fail pacemaker's cell-inversion check after reduction")
    train, test = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    train.to_pickle(out / "train.pckl.gzip", compression="gzip", protocol=PICKLE_PROTOCOL)
    test.to_pickle(out / "test.pckl.gzip", compression="gzip", protocol=PICKLE_PROTOCOL)
    io.write_json(out / "reference_energy.json", ref)
    # small per-structure table for the fit preflight (nfit vs DElow) and reports
    view = df[["natoms", "subsystem", "config_type", "fmax"]].copy()
    view["epa_corrected"] = df["energy_corrected"] / df["natoms"]
    if "hull_distance" in df:
        view["hull_distance"] = df["hull_distance"]
    view["test"] = is_test
    view.to_csv(out / "energetics.csv.gz", index=False)

    summary = {
        "n_total": len(df), "n_train": len(train), "n_test": len(test),
        "n_families": int(df["family"].nunique()),
        "reference_energy": ref,
        "by_subsystem": df.groupby("subsystem").size().to_dict(),
        "by_config_type": df.groupby("config_type").size().to_dict(),
        "energy_per_atom_range": [float((df.energy / df.natoms).min()), float((df.energy / df.natoms).max())],
        "max_natoms": int(df["natoms"].max()),
        "cells_reduced": reduced,
        **(extra_summary or {}),
    }
    io.write_json(out / "summary.json", summary)
    _plots(df, out)
    log.info("train %d / test %d structures (%d families)", len(train), len(test), summary["n_families"])
    io.mark_done(run_dir, "build_df", None, summary)


def _plots(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epa = df["energy"] / df["natoms"]
    vpa = df["ase_atoms"].map(lambda a: a.get_volume() / len(a))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].scatter(vpa, epa, s=2, alpha=0.4)
    axes[0].set(xlabel="volume per atom (Å³)", ylabel="teacher energy (eV/atom)", title="energy vs volume")
    axes[1].hist(df["natoms"], bins=np.arange(0.5, df["natoms"].max() + 1.5))
    axes[1].set(xlabel="atoms per cell", ylabel="structures", title="cell sizes")
    if "hull_distance" in df:
        axes[2].hist(df["hull_distance"], bins=100, log=True)
        axes[2].set(xlabel="distance above hull (eV/atom)", ylabel="structures", title="hull distance")
    else:
        axes[2].hist(epa, bins=100, log=True)
        axes[2].set(xlabel="energy (eV/atom)", ylabel="structures", title="energy")
    fig.tight_layout()
    fig.savefig(out / "dataset.png", dpi=130)
    plt.close(fig)
