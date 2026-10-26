"""Stage 2 (GRACE env): perturb, label with the teacher, filter.

Same recipe as ``generate_ternary_data.py`` (steps 6-8), plus:
  * seeded perturbations,
  * stress is kept (pacemaker does not fit it, but it is useful for checks),
  * ``drop_report`` counts every removed structure by subsystem and reason, so
    a filter that silently empties a corner (the W-rich liquid in NiAlW) shows.
"""

from __future__ import annotations

import logging
import time
from collections import Counter

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

from .. import io, recipe
from ..calculators import get_teacher

log = logging.getLogger(__name__)
STAGE_INDEX = 2


def _rounds(cfg, base) -> list:
    """(parent, rounds of the perturbation list) for the relaxed cells of this shard.

    Seeds get assyst.seed_copies rounds. Without reuse_from, assyst.boost adds `copies` rounds
    to cells within max_energy_above_ground of the lowest relaxed cell of the same formula
    (all cells of a random formula sit in one shard, so that minimum is complete).
    """
    a = cfg["assyst"]
    boost = a.get("boost") if not cfg["dataset"].get("reuse_from") else None
    lowest = {}
    if boost:
        for s in base:
            if "relax_epa" in s.info:
                key = s.get_chemical_formula(empirical=True)
                lowest[key] = min(lowest.get(key, np.inf), float(s.info["relax_epa"]))
    out = []
    for s in base:
        n = int(a.get("seed_copies", 1)) if str(s.info.get("origin", "")).startswith("seed:") else 1
        if boost and "relax_epa" in s.info:
            above = float(s.info["relax_epa"]) - lowest[s.get_chemical_formula(empirical=True)]
            if above <= float(boost["max_energy_above_ground"]):
                n += int(boost["copies"])
        out.append((s, n))
    return out


def run(cfg: dict, run_dir, shard: int):
    from assyst.filters import DistanceFilter
    from assyst.perturbations import perturb

    if io.is_done(run_dir, "label", shard):
        log.info("label shard %d already done", shard)
        return
    if not io.is_done(run_dir, "generate", shard):
        raise RuntimeError(f"generate shard {shard} is not done")

    a, flt = cfg["assyst"], cfg["filters"]
    rng = recipe.shard_rng(cfg["system"]["seed"], STAGE_INDEX, shard)
    relaxed = io.read_extxyz(io.stage_dir(run_dir, "generate") / f"{io.shard_name(shard)}.extxyz")
    last_step = a["relax"][-1]["type"] if a["relax"] else None
    # boost parents come from a reused dataset: perturbed here, but never labelled again
    boost_parents = [s for s in relaxed if s.info.get("step") == "boost_parent"]
    relaxed = [s for s in relaxed if s.info.get("step") != "boost_parent"]
    base = [s for s in relaxed if s.info.get("step") == last_step] if last_step else relaxed
    rounds = _rounds(cfg, base)
    for p in boost_parents:
        rounds.append((p, int(a["boost"]["copies"])))

    t0 = time.time()
    pipeline = recipe.build_perturbations(a["perturbations"], rng)
    dist = DistanceFilter(dict(a["distance_radii"]))
    perturbed = []
    for parent, n_rounds in rounds:
        for _ in range(n_rounds):
            for p in perturb([parent], pipeline, filters=[dist], retries=int(a["perturb_retries"])):
                p.info["parent"] = parent.info.get("uuid", "")
                p.info["step"] = "perturbed"
                perturbed.append(p)
    t_perturb = time.time() - t0
    log.info("%d relaxed (%d from last step), %d boost parents -> %d perturbed (%d rounds) in %.0f s",
             len(relaxed), len(base), len(boost_parents), len(perturbed), sum(n for _, n in rounds), t_perturb)

    calc = get_teacher(cfg)
    dropped = Counter()
    kept_by_subsystem = Counter()
    labelled = []
    t0 = time.time()
    for atoms in relaxed + perturbed:
        sub = io.subsystem(atoms)
        if atoms.get_volume() / len(atoms) > flt["max_volume_per_atom"]:
            dropped[(sub, "volume")] += 1
            continue
        try:
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = np.array(atoms.get_forces())
            try:
                stress = np.array(atoms.get_stress())
            except Exception:
                stress = None
        except Exception as ex:
            dropped[(sub, "teacher_error")] += 1
            log.debug("teacher failed: %s", ex)
            continue
        epa = e / len(atoms)
        if not (flt["min_energy_per_atom"] <= epa <= flt["max_energy_per_atom"]):
            dropped[(sub, "energy")] += 1
            continue
        if np.linalg.norm(f, axis=1).max() > flt["max_force"]:
            dropped[(sub, "force")] += 1
            continue
        atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f, stress=stress)
        io.clean_info(atoms)
        atoms.info["config_type"] = atoms.info.get("step", "unknown")
        labelled.append(atoms)
        kept_by_subsystem[sub] += 1
    t_label = time.time() - t0

    n = io.write_extxyz(io.stage_dir(run_dir, "label") / f"{io.shard_name(shard)}.extxyz", labelled)
    report = {
        "kept": n,
        "kept_by_subsystem": dict(kept_by_subsystem),
        "dropped": {f"{sub}|{why}": c for (sub, why), c in sorted(dropped.items())},
        "timings": {"perturb_s": t_perturb, "label_s": t_label},
    }
    io.write_json(io.stage_dir(run_dir, "label") / f"drop_report_{shard:03d}.json", report)
    log.info("kept %d, dropped %d, labelling %.0f s", n, sum(dropped.values()), t_label)
    io.mark_done(run_dir, "label", shard, report)
