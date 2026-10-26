"""Stage 1 (GRACE env): random symmetric crystals + relaxations with the teacher.

Same recipe as ``generate_ternary_data.py`` (steps 2 and 5), but sharded and
seeded. Writes ``generate/shard_XXX.extxyz`` holding the output of every
relaxation step (``info['step']`` says which).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
import warnings

from .. import io, recipe
from ..calculators import get_teacher

log = logging.getLogger(__name__)
STAGE_INDEX = 1


def _relax(settings, structure):
    """One relaxation; a run that hits max_steps is NOT an error.

    ASSYST >= 0.16 (commit 8f2f19e) raises NotConvergedError from Relax.relax() when
    force_tolerance is not reached within max_steps; calling the settings object
    instead returns the final structure with info['relax_converged'] = False.
    Older ASSYST (as used for test_05) returned the structure and did not check.
    """
    if callable(settings):
        return settings(structure)
    return settings.relax(structure)


def run(cfg: dict, run_dir, shard: int):
    from assyst.crystals import sample

    a = cfg["assyst"]
    n_shards = int(a["n_shards"])
    if io.is_done(run_dir, "generate", shard):
        log.info("generate shard %d already done", shard)
        return

    rng = recipe.shard_rng(cfg["system"]["seed"], STAGE_INDEX, shard)
    reuse = bool(cfg["dataset"].get("reuse_from"))
    random_crystals = a.get("random_crystals", True) and not reuse
    my_formulas = recipe.shard_formulas(recipe.formulas(cfg), shard, n_shards) if random_crystals else []
    my_seeds = recipe.build_seeds(cfg)[shard::n_shards]
    my_boost = recipe.boost_parents(cfg)[shard::n_shards] if reuse else []
    log.info("shard %d/%d: %d formulas, %d seeds, %d boost parents",
             shard, n_shards, len(my_formulas), len(my_seeds), len(my_boost))

    t0 = time.time()
    structures = list(my_seeds)
    for formula in my_formulas:
        groups = recipe.spacegroups(cfg, rng)
        structures.extend(sample(
            [formula],
            spacegroups=groups,
            min_atoms=1,
            max_atoms=max(sum(formula.values()), 1),
            max_structures=a["structures_per_formula"],
            tolerance=a["tolerance"],
            rng=rng,
        ))
    t_sample = time.time() - t0
    log.info("sampled %d structures in %.0f s", len(structures), t_sample)

    calc = get_teacher(cfg)
    kept, failed, reasons, timings = [], {}, {}, {"sample_s": t_sample}
    unconverged = {}
    current = structures
    for name, settings in recipe.build_relaxations(a["relax"]):
        t0 = time.time()
        relaxed = []
        for s in current:
            s = s.copy()
            s.calc = calc
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = _relax(settings, s)
            except Exception as ex:  # a bad cell must not kill the shard
                failed[name] = failed.get(name, 0) + 1
                reason = f"{type(ex).__name__}: {str(ex).splitlines()[0][:160] if str(ex) else ''}"
                reasons.setdefault(name, Counter())[reason] += 1
                if reasons[name][reason] == 1:   # every distinct reason once, with the cell
                    log.warning("relax %s failed for %s (%d atoms, %.1f A^3/atom, spacegroup %s): %s",
                                name, s.get_chemical_formula(), len(s), s.get_volume() / len(s),
                                s.info.get("spacegroup"), reason)
                continue
            try:   # kept for assyst.boost (label stage): the energy is not written otherwise
                r.info["relax_epa"] = float(r.get_potential_energy()) / len(r)
            except Exception:
                pass
            r.calc = None
            r.info["step"] = name
            if not r.info.get("relax_converged", True):
                unconverged[name] = unconverged.get(name, 0) + 1
                if a["drop_unconverged"]:
                    continue
            relaxed.append(r)
        timings[f"{name}_s"] = time.time() - t0
        log.info("%s: %d kept (%d not converged within max_steps%s), %d failed, %.0f s",
                 name, len(relaxed), unconverged.get(name, 0), ", dropped" if a["drop_unconverged"] else "",
                 failed.get(name, 0), timings[f"{name}_s"])
        kept.extend(relaxed)
        current = relaxed

    kept.extend(my_boost)          # already relaxed (reused dataset): perturbed in label, not relabelled
    for s in kept:
        io.clean_info(s)
    n = io.write_extxyz(io.stage_dir(run_dir, "generate") / f"{io.shard_name(shard)}.extxyz", kept)
    io.mark_done(run_dir, "generate", shard, {
        "formulas": len(my_formulas),
        "seeds": len(my_seeds),
        "boost_parents": len(my_boost),
        "sampled": len(structures),
        "written": n,
        "relax_unconverged": unconverged,
        "relax_failed": failed,
        "relax_failed_reasons": {k: dict(v) for k, v in reasons.items()},
        "timings": timings,
    })
