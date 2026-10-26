"""Run configuration: defaults, loading, validation.

Unknown keys are an ERROR, never silently ignored -- that is the exact trap
pacemaker has (``fit.loss.L1`` is swallowed without a word). The only
free-form sub-trees are ``fit.input`` (checked separately against
pacemaker's real keys, see :mod:`grace2ace.pacemaker_input`),
``teacher.params``, ``assyst.perturbations`` and ``slurm.<stage>.extra``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .stages import STAGES

# A key mapped to FREE accepts any value (dict or not) without key checking.
FREE = object()

DEFAULTS: dict[str, Any] = {
    "system": {
        "name": None,           # required; used for the run dir and the .yace name
        "elements": None,       # required
        "seed": 42,
        "run_dir": None,        # default: ./runs/<name>
    },
    "teacher": {
        "calculator": "grace",  # grace | morse
        "model": "GRACE-2L-SMAX-OMAT-large",
        "params": FREE,         # extra keyword arguments for the calculator
    },
    "assyst": {
        # Formulas.range(elements, *range) then .trim(*trim); same meaning as
        # in the old generate_ternary_data.py scripts.
        "range": [0, 9],
        "trim": [1, 10],
        "spacegroups": "all",   # "all" or a list of ints
        "shuffle_spacegroups": True,
        "structures_per_formula": 230,
        "tolerance": "metallic",
        "relax": [
            {"type": "VolumeRelax", "max_steps": 200, "force_tolerance": 1.0e-3},
            {"type": "FullRelax", "max_steps": 200, "force_tolerance": 1.0e-3},
        ],
        # a relaxation that reaches max_steps before force_tolerance: keep its final
        # structure (default; what test_05's older ASSYST did) or drop it
        "drop_unconverged": False,
        "perturbations": FREE,
        "perturb_retries": 5,
        # random symmetric crystals from pyxtal (the classic ASSYST data). With dataset.reuse_from
        # no random crystals are made: the reused dataset already has them.
        "random_crystals": True,
        # known structures, relaxed and perturbed like the random ones; see recipe.build_seeds:
        #   - {name: Si_diamond, bulk: {name: Si, crystalstructure: diamond, a: 5.465, cubic: true},
        #      supercells: [[1,1,1],[2,2,2]], substitutions: [{from: Si, to: Al, counts: [1], samples: 2}]}
        #   - {name: Si_BC8, crystal: {symbols: [Si], basis: [[0.1003,0.1003,0.1003]], spacegroup: 206,
        #      cellpar: [6.64,6.64,6.64,90,90,90]}}
        #   - {name: mine, file: /path/structures.extxyz}
        "seeds": FREE,
        "seed_copies": 1,       # rounds of the perturbation list per relaxed seed
        # extra perturbation rounds for relaxed cells close to their ground state:
        #   {max_energy_above_ground: 0.1, copies: 3}
        # "ground" = the convex hull of the reused dataset (with reuse_from; its relaxed cells are
        # the parents), else the lowest relaxed cell of the same formula in the shard
        "boost": FREE,
        "distance_radii": None,  # {El: radius}; required
        "n_shards": 1,
        # executor local / slurm_single: how many shards run AT THE SAME TIME on the one
        # GPU (small cells leave the GPU mostly idle, so several share it well)
        "parallel_shards": 1,
    },
    "filters": {
        "max_volume_per_atom": 40.0,
        "min_energy_per_atom": -15.0,
        "max_energy_per_atom": 5.0,
        "max_force": 20.0,
    },
    "extra_data": [],           # list of {path, name, format, group_column}
    "dataset": {
        "test_fraction": 0.1,
        "reference_energy": "freeze",  # "freeze" or {El: e, ..., shift: s}
        # another run directory whose finished dataset (build_df/) this run fits on:
        # generate/label/build_df are skipped and the same train/test sets and
        # reference energy are used, so the two potentials are directly comparable
        "reuse_from": None,
        # with reuse_from: build a FILTERED copy of that dataset instead of using it as is.
        # Keys as in `filters` (e.g. {max_volume_per_atom: 50}); the train/test split and the
        # reference energy of the source are kept, so the runs stay comparable.
        "reuse_filters": None,
    },
    "fit": {
        "input": FREE,          # deep-merged over templates/pacemaker_default.yaml
        "max_attempts": 5,
        "timeout_margin_minutes": 20,   # stop pacemaker this long before the job's walltime
        "min_start_minutes": 30,        # do not start a fit attempt with less time than this
        "local_timeout_hours": None,
        # where the ZBL wall starts after the fit:
        #   auto         pacemaker's `repulsion: auto` = shortest distance per bond in the fitted data
        #   pace_corerep pacemaker's pace_corerep tool: compress small cells and put the wall where
        #                ACE stops being reliable (energy kink or extrapolation grade > 10)
        "core_repulsion": "auto",
    },
    "evaluate": {
        "max_train_structures": 5000,
        "hull_bins": [0.05, 0.2, 0.5, 1.0, 1.5],
    },
    "envs": {
        "generate": None,       # python of the GRACE/ASSYST env
        "fit": None,            # python of the pacemaker env
    },
    "executor": "slurm",        # slurm | slurm_single | local
    "slurm": {
        "account": None,
        "setup": [],            # shell lines run before every stage
        **{
            stage: {
                "partition": None,
                "time": "24:00:00",
                "gpus": 0,
                "cpus": 4,
                "mem": None,
                "extra": FREE,  # extra "#SBATCH ..." argument strings
            }
            for stage in (*STAGES, "single")   # "single": the one job of executor slurm_single
        },
    },
}

# slurm:        one job per stage, chained by dependencies, shards in parallel
# slurm_single: ONE job (slurm.single) runs every stage in turn and resubmits itself
#               if the fit has to stop at the walltime
# local:        everything here, in this shell
EXECUTORS = ("slurm", "slurm_single", "local")

REQUIRED = [
    ("system", "name"),
    ("system", "elements"),
    ("assyst", "distance_radii"),
    ("envs", "generate"),
    ("envs", "fit"),
]


class ConfigError(ValueError):
    pass


def _strip_free(d):
    """Defaults with FREE markers replaced by sensible empty values."""
    if d is FREE:
        return None
    if isinstance(d, dict):
        return {k: _strip_free(v) for k, v in d.items()}
    return copy.deepcopy(d)


def _check_and_merge(defaults, user, path=""):
    """Merge ``user`` over ``defaults``, raising on keys not in ``defaults``."""
    if defaults is FREE:
        return copy.deepcopy(user)
    if isinstance(defaults, dict) and not isinstance(user, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping with keys {sorted(defaults)}, got {user!r}")
    if not isinstance(defaults, dict):
        return copy.deepcopy(user)
    out = _strip_free(defaults)
    for key, value in user.items():
        if key not in defaults:
            where = path or "<top level>"
            raise ConfigError(
                f"unknown config key '{key}' in {where}; "
                f"allowed: {sorted(defaults)}"
            )
        out[key] = _check_and_merge(defaults[key], value, f"{path}.{key}" if path else key)
    return out


def _expand(p):
    return None if p is None else os.path.abspath(os.path.expanduser(os.path.expandvars(str(p))))


def validate(cfg: dict) -> dict:
    """Merge with defaults, check required keys and simple consistency."""
    cfg = _check_and_merge(DEFAULTS, cfg or {})

    for section, key in REQUIRED:
        if cfg[section][key] in (None, "", []):
            raise ConfigError(f"missing required config key {section}.{key}")

    elements = list(cfg["system"]["elements"])
    cfg["system"]["elements"] = elements

    missing = [e for e in elements if e not in cfg["assyst"]["distance_radii"]]
    if missing:
        raise ConfigError(f"assyst.distance_radii has no radius for {missing}")

    if not cfg["assyst"]["perturbations"]:
        cfg["assyst"]["perturbations"] = []
    if int(cfg["assyst"]["n_shards"]) < 1:
        raise ConfigError("assyst.n_shards must be >= 1")
    if int(cfg["assyst"]["parallel_shards"]) < 1:
        raise ConfigError("assyst.parallel_shards must be >= 1")

    if cfg["executor"] not in EXECUTORS:
        raise ConfigError(f"executor must be one of {EXECUTORS}, not {cfg['executor']!r}")

    if cfg["teacher"]["calculator"] not in ("grace", "morse"):
        raise ConfigError("teacher.calculator must be 'grace' or 'morse'")

    ref = cfg["dataset"]["reference_energy"]
    if not (ref == "freeze" or isinstance(ref, dict)):
        raise ConfigError("dataset.reference_energy must be 'freeze' or a dict {El: e, shift: s}")

    for i, extra in enumerate(cfg["extra_data"]):
        unknown = set(extra) - {"path", "name", "format", "group_column"}
        if unknown or "path" not in extra:
            raise ConfigError(
                f"extra_data[{i}] needs 'path' and may have name/format/group_column; got {sorted(extra)}"
            )
        extra["path"] = _expand(extra["path"])
        extra.setdefault("name", Path(extra["path"]).stem)
        extra.setdefault("format", "extxyz" if extra["path"].endswith((".xyz", ".extxyz")) else "pacemaker")
        extra.setdefault("group_column", None)

    for key in ("generate", "fit"):
        cfg["envs"][key] = _expand(cfg["envs"][key])

    if cfg["dataset"]["reuse_from"]:
        cfg["dataset"]["reuse_from"] = _expand(cfg["dataset"]["reuse_from"])

    if cfg["system"]["run_dir"] is None:
        cfg["system"]["run_dir"] = os.path.join("runs", cfg["system"]["name"])
    cfg["system"]["run_dir"] = _expand(cfg["system"]["run_dir"])

    if cfg["dataset"]["reuse_from"] and cfg["dataset"]["reuse_from"] == cfg["system"]["run_dir"]:
        raise ConfigError("dataset.reuse_from points at this run's own run_dir")

    rf = cfg["dataset"]["reuse_filters"]
    if rf:
        if not cfg["dataset"]["reuse_from"]:
            raise ConfigError("dataset.reuse_filters needs dataset.reuse_from")
        if not isinstance(rf, dict) or set(rf) - set(DEFAULTS["filters"]):
            raise ConfigError(f"dataset.reuse_filters takes keys {sorted(DEFAULTS['filters'])}, got {rf!r}")
        cfg["dataset"]["reuse_filters"] = {k: float(v) for k, v in rf.items()}

    a = cfg["assyst"]
    a["seeds"] = a["seeds"] or []
    for i, sd in enumerate(a["seeds"]):
        kinds = [k for k in ("bulk", "crystal", "file") if k in sd]
        unknown = set(sd) - {"name", "bulk", "crystal", "file", "supercells", "substitutions"}
        if len(kinds) != 1 or unknown:
            raise ConfigError(f"assyst.seeds[{i}] needs exactly one of bulk/crystal/file "
                              f"(optional: name, supercells, substitutions); got {sorted(sd)}")
        for sub in sd.get("substitutions") or []:
            if set(sub) - {"from", "to", "counts", "samples"} or not {"from", "to", "counts"} <= set(sub):
                raise ConfigError(f"assyst.seeds[{i}].substitutions: use from, to, counts[, samples]; got {sub}")
    if int(a["seed_copies"]) < 1:
        raise ConfigError("assyst.seed_copies must be >= 1")
    if a["boost"]:
        b = a["boost"]
        if set(b) != {"max_energy_above_ground", "copies"} or int(b["copies"]) < 1:
            raise ConfigError("assyst.boost needs {max_energy_above_ground: <eV/atom>, copies: <int >= 1>}")
    if not cfg["dataset"]["reuse_from"] and not a["random_crystals"] and not a["seeds"]:
        raise ConfigError("nothing to generate: assyst.random_crystals is false and there are no seeds")

    if cfg["fit"]["core_repulsion"] not in ("auto", "pace_corerep"):
        raise ConfigError("fit.core_repulsion must be 'auto' or 'pace_corerep'")

    if cfg["fit"]["input"] is None:
        cfg["fit"]["input"] = {}
    if cfg["teacher"]["params"] is None:
        cfg["teacher"]["params"] = {}
    for stage in (*STAGES, "single"):
        if cfg["slurm"][stage]["extra"] is None:
            cfg["slurm"][stage]["extra"] = []
    return cfg


def load(path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return validate(raw)


def load_frozen(run_dir) -> dict:
    """Load the config copy stored inside a run directory.

    It is validated again, so settings added to grace2ace after the run was
    created get their defaults instead of a KeyError on resume.
    """
    with open(Path(run_dir) / "config.yaml") as f:
        return validate(yaml.safe_load(f))


def freeze(cfg: dict, run_dir) -> Path:
    path = Path(run_dir) / "config.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
