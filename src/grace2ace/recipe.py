"""Turn the ``assyst`` config section into ASSYST objects.

Perturbations are written in YAML and map 1:1 onto ASSYST classes::

    perturbations:
      - stretch: {hydro: 0.15, shear: 0.02}            # Stretch(hydro=0.15, shear=0.02)
      - rattle: {sigma: 0.03, create_supercells: true}  # Rattle(sigma=0.03, create_supercells=True)
      - [ {stretch: {hydro: 0.05, shear: 0.005}},       # a list is a Series:
          {rattle: {sigma: 0.05}} ]                     #   Stretch(...) + Rattle(...)
      - random_choice:                                  # RandomChoice(a, b, chance)
          a: {rattle: {sigma: 0.25}}
          b: {stretch: {hydro: 0.4, shear: 0.05}}
          chance: 0.3                                   #   probability of picking b
      - repeat: 4                                       # optional: the next item 4 times
        item: {stretch: {hydro: 0.8, shear: 0.05}}
"""

from __future__ import annotations

import numpy as np

_SIMPLE = {
    "stretch": "Stretch",
    "rattle": "Rattle",
    "element_scaled_rattle": "ElementScaledRattle",
}


def build_perturbation(spec, rng):
    """One YAML item -> one ASSYST perturbation."""
    from assyst import perturbations as P

    if isinstance(spec, list):
        if not spec:
            raise ValueError("empty perturbation series")
        parts = [build_perturbation(s, rng) for s in spec]
        return P.Series(tuple(parts))
    if not isinstance(spec, dict) or len(spec) != 1:
        raise ValueError(f"perturbation must be a one-key mapping or a list, got {spec!r}")
    (kind, args), = spec.items()
    args = dict(args or {})
    if kind in _SIMPLE:
        return getattr(P, _SIMPLE[kind])(**args, rng=rng)
    if kind == "random_choice":
        unknown = set(args) - {"a", "b", "chance"}
        if unknown:
            raise ValueError(f"random_choice takes a, b, chance; got extra {sorted(unknown)}")
        return P.RandomChoice(
            build_perturbation(args["a"], rng),
            build_perturbation(args["b"], rng),
            float(args["chance"]),
            rng=rng,
        )
    raise ValueError(f"unknown perturbation {kind!r}; use one of {sorted(_SIMPLE) + ['random_choice']}")


def build_perturbations(specs, rng) -> list:
    """The full YAML list -> flat list of perturbations (``repeat`` expanded)."""
    out = []
    for spec in specs or []:
        if isinstance(spec, dict) and "repeat" in spec:
            unknown = set(spec) - {"repeat", "item"}
            if unknown:
                raise ValueError(f"repeat takes repeat, item; got extra {sorted(unknown)}")
            out.extend(build_perturbation(spec["item"], rng) for _ in range(int(spec["repeat"])))
        else:
            out.append(build_perturbation(spec, rng))
    return out


def build_relaxations(specs) -> list:
    from assyst import relaxations as R

    allowed = {"Relax", "CellRelax", "VolumeRelax", "SymmetryRelax", "FullRelax"}
    out = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type")
        if kind not in allowed:
            raise ValueError(f"unknown relaxation {kind!r}; use one of {sorted(allowed)}")
        out.append((kind, getattr(R, kind)(**spec)))
    return out


def formulas(cfg) -> list[dict]:
    """All formulas, in the same order as the old scripts."""
    from assyst.crystals import Formulas

    a = cfg["assyst"]
    f = Formulas.range(tuple(cfg["system"]["elements"]), *a["range"]).trim(*a["trim"])
    # drop zero counts (pyxtal never returns structures for them) and empties
    out = []
    for formula in f:
        formula = {e: int(n) for e, n in formula.items() if n > 0}
        if formula:
            out.append(formula)
    return out


def shard_formulas(all_formulas, shard, n_shards):
    """Round-robin split, so every shard gets a similar mix of small and large cells."""
    return all_formulas[shard::n_shards]


def shard_rng(seed, stage_index, shard):
    """Independent, reproducible random stream per (stage, shard)."""
    return np.random.default_rng([int(seed), int(stage_index), int(shard)])


def spacegroups(cfg, rng) -> list[int]:
    sg = cfg["assyst"]["spacegroups"]
    groups = list(range(1, 231)) if sg == "all" else [int(g) for g in sg]
    if cfg["assyst"]["shuffle_spacegroups"]:
        groups = [int(g) for g in rng.permutation(groups)]
    return groups


# ---- known structures (assyst.seeds) and boost parents ------------------------------

def _seed_base(spec):
    """One seed spec -> list of ase.Atoms (a file may hold several)."""
    if "bulk" in spec:
        from ase.build import bulk

        kw = dict(spec["bulk"])
        return [bulk(kw.pop("name"), **kw)]
    if "crystal" in spec:
        from ase.spacegroup import crystal

        kw = dict(spec["crystal"])
        return [crystal(**kw)]
    import ase.io

    return list(ase.io.read(spec["file"], index=":"))


def _substituted(atoms, sub, rng):
    """Random substitutions of `from` atoms by `to`, `samples` cells per count."""
    out = []
    sites = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == sub["from"]]
    for count in sub["counts"]:
        if count > len(sites):
            continue
        for k in range(int(sub.get("samples", 1))):
            a = atoms.copy()
            for i in rng.choice(sites, size=int(count), replace=False):
                a[int(i)].symbol = sub["to"]
            a.info["seed_variant"] = f"{sub['from']}->{sub['to']} x{count} #{k}"
            out.append(a)
    return out


def build_seeds(cfg) -> list:
    """All seed structures, in a fixed order (sharded by index like formulas).

    Each spec gives its base cell(s); each supercell of each base is a seed, and so is
    every random substitution of it. Every seed is its own family (train/test split).
    """
    from assyst.utils import update_uuid

    rng = np.random.default_rng([int(cfg["system"]["seed"]), 99])
    seeds = []
    for i, spec in enumerate(cfg["assyst"].get("seeds") or []):
        name = spec.get("name", f"seed{i}")
        for base in _seed_base(spec):
            base.pbc = True
            for rep in spec.get("supercells") or [[1, 1, 1]]:
                cell = base.repeat(tuple(int(r) for r in rep))
                variants = [cell]
                for sub in spec.get("substitutions") or []:
                    variants += _substituted(cell, sub, rng)
                for v in variants:
                    v.info.clear()
                    v.info["origin"] = f"seed:{name}"
                    update_uuid(v)                      # uuid; info['seed'] -> family
                    seeds.append(v)
    return seeds


def boost_parents(cfg) -> list:
    """Relaxed cells of the reused dataset within max_energy_above_ground of its hull.

    They are perturbed again (assyst.boost.copies rounds) but not re-labelled themselves.
    Each keeps its family and its train/test side, so its copies cannot leak into the
    other set.
    """
    import pandas as pd

    from . import io

    b = cfg["assyst"].get("boost")
    src = cfg["dataset"].get("reuse_from")
    if not b or not src:
        return []
    last = cfg["assyst"]["relax"][-1]["type"]
    parents = []
    for split in ("train", "test"):
        df = pd.read_pickle(io.stage_dir(src, "build_df") / f"{split}.pckl.gzip", compression="gzip")
        df = df[(df["config_type"] == last) & (df["hull_distance"] <= float(b["max_energy_above_ground"]))]
        for _, r in df.iterrows():
            a = r["ase_atoms"].copy()
            a.calc = None
            a.info = {"uuid": str(r["name"]).rsplit("/", 1)[-1], "family": r["family"], "step": "boost_parent",
                      "origin": "boost", "split": split}
            parents.append(a)
    parents.sort(key=lambda a: a.info["uuid"])       # fixed order, independent of the source layout
    return parents
