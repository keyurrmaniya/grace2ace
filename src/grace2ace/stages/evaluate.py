"""Stage 5 (pacemaker env): how well did the student learn?

Predicts the WHOLE leak-free test set (not only pacemaker's nfit subsample)
with the fitted potential and reports errors binned by distance above the
convex hull -- the headline RMSE is dominated by high-energy structures, the
near-hull bins are what matters for thermodynamics. Writes
``evaluate/report.json`` and ``evaluate/parity.png``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import io

log = logging.getLogger(__name__)


def predict(df, potential_file):
    from pyace import PyACECalculator

    calc = PyACECalculator(str(potential_file))
    e_pred, f_pred = [], []
    for at in df["ase_atoms"]:
        at = at.copy()
        at.calc = calc
        e_pred.append(float(at.get_potential_energy()))
        f_pred.append(np.array(at.get_forces()))
    df = df.copy()
    df["energy_pred"] = e_pred
    df["forces_pred"] = f_pred
    return df


def errors(df) -> pd.DataFrame:
    """Per-structure error columns (meV/atom, meV/Å)."""
    df = df.copy()
    df["dE_meV_at"] = 1000 * (df["energy_pred"] - df["energy_corrected"]) / df["natoms"]
    df["dF_sq_sum"] = [float(np.sum((np.asarray(p) - np.asarray(t)) ** 2))
                       for p, t in zip(df["forces_pred"], df["forces"])]
    df["dF_ncomp"] = 3 * df["natoms"]
    return df


def summarise(df) -> dict:
    if len(df) == 0:
        return {"N": 0}
    return {
        "N": int(len(df)),
        "E_rmse_meV_at": float(np.sqrt(np.mean(df["dE_meV_at"] ** 2))),
        "E_mae_meV_at": float(np.mean(np.abs(df["dE_meV_at"]))),
        "F_rmse_meV_A": float(1000 * np.sqrt(df["dF_sq_sum"].sum() / df["dF_ncomp"].sum())),
    }


def report_for(df, bins) -> dict:
    out = {"overall": summarise(df)}
    if "hull_distance" in df:
        edges = [-1e-9] + list(bins) + [np.inf]
        labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}" for lo, hi in zip([0] + list(bins), edges[1:])]
        cut = pd.cut(df["hull_distance"], edges, labels=labels)
        out["by_hull_distance_eV_at"] = {str(lab): summarise(df[cut == lab]) for lab in labels}
    out["by_subsystem"] = {k: summarise(g) for k, g in df.groupby("subsystem")}
    out["by_config_type"] = {k: summarise(g) for k, g in df.groupby("config_type")}
    sq = np.sort(df["dE_meV_at"].to_numpy() ** 2)
    if len(sq):
        top = max(1, int(0.05 * len(sq)))
        out["top5pct_share_of_squared_E_error"] = float(sq[-top:].sum() / sq.sum()) if sq.sum() > 0 else 0.0
    return out


def run(cfg: dict, run_dir, shard=None):
    if io.is_done(run_dir, "evaluate"):
        return
    if not io.is_done(run_dir, "fit"):
        raise RuntimeError("fit is not done")
    name = cfg["system"]["name"]
    pot = io.stage_dir(run_dir, "fit") / f"{name}.yaml"
    data = io.data_dir(cfg, run_dir)
    bins = cfg["evaluate"]["hull_bins"]

    test = errors(predict(pd.read_pickle(data / "test.pckl.gzip", compression="gzip"), pot))
    train = pd.read_pickle(data / "train.pckl.gzip", compression="gzip")
    n_train = int(cfg["evaluate"]["max_train_structures"])
    if len(train) > n_train:
        train = train.sample(n=n_train, random_state=int(cfg["system"]["seed"]))
    train = errors(predict(train, pot))

    report = {
        "dataset": str(data),
        "potential": str(pot),
        "yace": str(pot.with_suffix(".yace")),
        "test": report_for(test, bins),
        "train_sample": report_for(train, bins),
        "fit": io.read_json(io.stage_dir(run_dir, "fit") / "status.json"),
    }
    out = io.stage_dir(run_dir, "evaluate")
    io.write_json(out / "report.json", report)
    keep = [c for c in ("name", "subsystem", "config_type", "natoms", "hull_distance", "dE_meV_at") if c in test]
    test[keep].to_csv(out / "test_errors.csv.gz", index=False)
    _parity(test, out)
    o = report["test"]["overall"]
    log.info("test: E %.1f meV/at, F %.1f meV/A over %d structures",
             o["E_rmse_meV_at"], o["F_rmse_meV_A"], o["N"])
    io.mark_done(run_dir, "evaluate", None, {"test_overall": o})


def _parity(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    e_t = df["energy_corrected"] / df["natoms"]
    e_p = df["energy_pred"] / df["natoms"]
    color = df["hull_distance"] if "hull_distance" in df else None
    sc = axes[0].scatter(e_t, e_p, s=3, c=color, cmap="viridis", alpha=0.6)
    lim = [min(e_t.min(), e_p.min()), max(e_t.max(), e_p.max())]
    axes[0].plot(lim, lim, "k--", lw=0.8)
    axes[0].set(xlabel="teacher energy (eV/atom)", ylabel="ACE energy (eV/atom)",
                title="energy, test set (colour = eV/atom above hull)")
    if color is not None:
        fig.colorbar(sc, ax=axes[0])
    f_t = np.concatenate([np.ravel(f) for f in df["forces"]])
    f_p = np.concatenate([np.ravel(f) for f in df["forces_pred"]])
    axes[1].scatter(f_t, f_p, s=1, alpha=0.3)
    lim = [min(f_t.min(), f_p.min()), max(f_t.max(), f_p.max())]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set(xlabel="teacher force component (eV/Å)", ylabel="ACE force component (eV/Å)",
                title="forces, test set")
    fig.tight_layout()
    fig.savefig(out / "parity.png", dpi=130)
    plt.close(fig)
