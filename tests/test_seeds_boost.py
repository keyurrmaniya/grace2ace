import numpy as np
import pytest
from ase.build import bulk

from grace2ace import config, io, recipe
from grace2ace.stages.label import _rounds


def _cfg(assyst=None, dataset=None):
    a = {"distance_radii": {"Si": 1.0, "Al": 1.0}}
    a.update(assyst or {})
    return config.validate({"system": {"name": "t", "elements": ["Si", "Al"], "run_dir": "/tmp/g2a_s"},
                            "assyst": a, "envs": {"generate": "/x", "fit": "/y"}, "dataset": dataset or {}})


SEEDS = [{"name": "dia", "bulk": {"name": "Si", "crystalstructure": "diamond", "a": 5.465, "cubic": True},
          "supercells": [[1, 1, 1], [2, 1, 1]],
          "substitutions": [{"from": "Si", "to": "Al", "counts": [1, 2], "samples": 3}]},
         {"name": "bc8", "crystal": {"symbols": ["Si"], "basis": [[0.1003, 0.1003, 0.1003]], "spacegroup": 206,
                                     "cellpar": [6.64, 6.64, 6.64, 90, 90, 90]}}]


def test_build_seeds_counts_and_families():
    seeds = recipe.build_seeds(_cfg({"seeds": SEEDS}))
    # diamond: 2 supercells x (1 pure + 2 counts x 3 samples) = 14; BC8: 1 cell of 16 atoms
    assert len(seeds) == 15
    assert [len(s) for s in seeds[:2]] == [8, 8] and len(seeds[7]) == 16 and len(seeds[-1]) == 16
    assert sum(s.get_chemical_symbols().count("Al") for s in seeds[1:4]) == 3
    assert len({s.info["seed"] for s in seeds}) == 15            # every seed its own family
    assert all(s.info["origin"].startswith("seed:") for s in seeds)
    again = recipe.build_seeds(_cfg({"seeds": SEEDS}))
    assert [s.get_chemical_symbols() for s in again] == [s.get_chemical_symbols() for s in seeds]  # reproducible


def test_seed_and_boost_validation():
    with pytest.raises(config.ConfigError, match="exactly one of bulk/crystal/file"):
        _cfg({"seeds": [{"bulk": {"name": "Si"}, "file": "x"}]})
    with pytest.raises(config.ConfigError, match="assyst.boost"):
        _cfg({"boost": {"copies": 2}})
    with pytest.raises(config.ConfigError, match="nothing to generate"):
        _cfg({"random_crystals": False})


def test_what_a_run_generates():
    assert io.generates_structures(_cfg())                                       # fresh: random crystals
    plain = _cfg(dataset={"reuse_from": "/tmp/other"})
    assert not io.generates_structures(plain) and not io.reuse_copy(plain)       # plain reuse as before
    seeded = _cfg({"seeds": SEEDS}, {"reuse_from": "/tmp/other"})
    assert io.generates_structures(seeded) and io.reuse_copy(seeded)
    assert io.dataset_source(seeded, "/tmp/g2a_s").as_posix() == "/tmp/g2a_s"
    old_frozen = {"dataset": {"reuse_from": "/tmp/other"}, "assyst": {"n_shards": 4}}   # e.g. NiAlW_01
    assert not io.generates_structures(old_frozen) and not io.reuse_copy(old_frozen)


def test_rounds_for_seeds_and_boost():
    cfg = _cfg({"seeds": SEEDS, "seed_copies": 3, "boost": {"max_energy_above_ground": 0.1, "copies": 2}})
    low, high, seed = bulk("Si", "diamond", a=5.4), bulk("Si", "diamond", a=5.6), bulk("Al", "fcc", a=4.0)
    low.info["relax_epa"], high.info["relax_epa"] = -5.40, -5.20
    seed.info.update(origin="seed:x", relax_epa=-3.7)
    r = dict((id(s), n) for s, n in _rounds(cfg, [low, high, seed]))
    assert r[id(low)] == 1 + 2          # near its formula's lowest cell: boosted
    assert r[id(high)] == 1             # 0.2 eV/atom above: not boosted
    assert r[id(seed)] == 3 + 2         # seed copies, and it is the lowest Al cell
