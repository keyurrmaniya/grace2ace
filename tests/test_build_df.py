import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from grace2ace.stages import build_df
from grace2ace.stages.fit import walltime_seconds, last_progress
from grace2ace.executors import _array_spec


def toy_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        at = bulk("Ni", "fcc", a=3.5 + 0.1 * rng.random(), cubic=True)
        at.symbols[: rng.integers(0, 4)] = "Al"
        rows.append({"ase_atoms": at, "family": f"f{i // 3}"})
    df = pd.DataFrame(rows)
    df["natoms"] = df["ase_atoms"].map(len)
    comp = build_df.composition_matrix(df["ase_atoms"], ["Ni", "Al"])
    df["energy"] = comp @ np.array([-5.0, -3.0]) + rng.normal(0, 0.05, n)
    return df


def test_freeze_reference_recovers_element_energies():
    df = toy_df()
    ref = build_df.freeze_reference(df, ["Ni", "Al"])
    assert ref["Ni"] == pytest.approx(-5.0, abs=0.05)
    assert ref["Al"] == pytest.approx(-3.0, abs=0.05)
    build_df.apply_reference(df, ["Ni", "Al"], ref)
    vpa = df["ase_atoms"].map(lambda a: a.get_volume() / len(a))
    # pacemaker's convention: the largest-volume structure sits at zero
    assert df["energy_corrected"][vpa.idxmax()] == pytest.approx(0.0, abs=1e-10)


def test_freeze_matches_pyace_when_available():
    process_df = pytest.importorskip("pyace.process_df")
    df = toy_df()
    ours = build_df.freeze_reference(df, ["Al", "Ni"])
    theirs_df = df[["ase_atoms", "energy"]].copy()
    theirs_df["NUMBER_OF_ATOMS"] = df["natoms"]
    theirs = process_df.compute_shifted_scaled_corrected_energy(theirs_df)
    for k in ("Ni", "Al", "shift"):
        assert ours[k] == pytest.approx(theirs[k], abs=1e-8)


def test_family_split_has_no_leak():
    df = toy_df(90)
    is_test = build_df.family_split(df, 0.2, seed=3)
    assert 0.15 < is_test.mean() < 0.3
    assert not set(df["family"][is_test]) & set(df["family"][~is_test])


def test_walltime_parsing():
    assert walltime_seconds("48:00:00") == 48 * 3600
    assert walltime_seconds("1-02:00:00") == 26 * 3600
    assert walltime_seconds("30") == 30 * 60
    assert walltime_seconds("05:30") == 5 * 60 + 30


def test_array_spec():
    assert _array_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert _array_spec([3]) == "3"


def test_last_progress(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "2026-09-05 18:58:41,514 I - Iteration   #498  (502 evals):      Loss: 4.8652e-03 | RMSE ...\n"
        "Iteration:  #500 Loss:    Total:  6.0767e-03 (100%) \n"
    )
    p = last_progress(log)
    assert p["iteration"] == 500 and p["loss"] == pytest.approx(6.0767e-3)
