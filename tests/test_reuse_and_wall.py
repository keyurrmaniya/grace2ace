import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from grace2ace import config, io
from grace2ace.stages.build_df import filter_mask
from grace2ace.stages.fit import wall_start


def _cfg(**dataset):
    return config.validate({
        "system": {"name": "t", "elements": ["Ni"], "run_dir": "/tmp/g2a_t"},
        "assyst": {"distance_radii": {"Ni": 1.0}},
        "envs": {"generate": "/x", "fit": "/y"},
        "dataset": dataset,
    })


def test_filter_mask_volume_and_force():
    cells = [bulk("Ni", "fcc", a=a) for a in (3.5, 4.5, 6.0)]          # 10.7, 22.8, 54 A^3/atom
    df = pd.DataFrame({"ase_atoms": cells, "natoms": [1, 1, 1],
                       "energy": [-5.0, -4.0, -1.0], "fmax": [0.1, 30.0, 0.1]})
    keep, why = filter_mask(df, {"max_volume_per_atom": 50.0, "max_force": 20.0})
    assert keep.tolist() == [True, False, False]
    assert why == {"max_volume_per_atom": 1, "max_force": 1}


def test_reuse_filters_need_reuse_from_and_known_keys():
    with pytest.raises(config.ConfigError, match="needs dataset.reuse_from"):
        _cfg(reuse_filters={"max_volume_per_atom": 50})
    with pytest.raises(config.ConfigError, match="takes keys"):
        _cfg(reuse_from="/tmp/other", reuse_filters={"max_vol": 50})
    cfg = _cfg(reuse_from="/tmp/other", reuse_filters={"max_volume_per_atom": 50})
    assert io.reuse_filtered(cfg)
    # a filtered reuse builds its OWN dataset; a plain reuse reads the other run's
    assert io.dataset_source(cfg, "/tmp/g2a_t").as_posix() == "/tmp/g2a_t"
    plain = _cfg(reuse_from="/tmp/other")
    assert not io.reuse_filtered(plain)
    assert io.dataset_source(plain, "/tmp/g2a_t").as_posix() == "/tmp/other"


def test_old_frozen_configs_without_new_keys_still_work():
    old = {"dataset": {"reuse_from": "/tmp/other"}}            # e.g. NiAlW_01's frozen config
    assert not io.reuse_filtered(old)
    assert io.dataset_source(old, "/tmp/x").as_posix() == "/tmp/other"


def test_wall_start_reads_pair_blocks_only(tmp_path):
    p = tmp_path / "pot.yaml"
    p.write_text("species:\n  - speciesblock: Bi\n    r_in: 2.31\n  - speciesblock: Bi Pb\n    r_in: 2.29\n"
                 "  - speciesblock: Bi Pb Sn\n    r_in: 9.9\n")
    assert wall_start(p) == {"Bi": 2.31, "Bi-Pb": 2.29}


def test_core_repulsion_value_checked():
    with pytest.raises(config.ConfigError, match="core_repulsion"):
        config.validate({"system": {"name": "t", "elements": ["Ni"]}, "assyst": {"distance_radii": {"Ni": 1.0}},
                         "envs": {"generate": "/x", "fit": "/y"}, "fit": {"core_repulsion": "zbl"}})


def test_needle_cell_is_reduced_without_changing_labels():
    from ase.calculators.emt import EMT
    a = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
    # same lattice, written with a long sheared vector: cell2 = cell0 + 9*cell1 (+ shear)
    cell = a.cell[:].copy(); cell[0] = cell[0] + 9 * cell[1]; cell[2] = cell[2] - 7 * cell[1]
    a.set_cell(cell, scale_atoms=False); a.wrap()
    assert np.linalg.cond(a.cell[:]) > io.MAX_CELL_CONDITION
    b = a.copy()
    assert io.reduce_cell(b)
    assert np.linalg.cond(b.cell[:]) < 5 and io.pacemaker_cell_ok(b.cell[:])
    a.calc = EMT(); b.calc = EMT()
    assert abs(a.get_potential_energy() - b.get_potential_energy()) < 1e-9
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-9      # Cartesian frame kept


def test_good_cell_left_alone():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    before = a.cell[:].copy()
    assert not io.reduce_cell(a) and np.allclose(a.cell[:], before)
