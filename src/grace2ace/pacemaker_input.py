"""Build and check the pacemaker input file.

The key lists below were read from pyace 0.2.7's source (``generalfit.py``,
``preparedata.py``, ``lossfuncspec.py``, ``multispecies_basisextension.py``,
``fitadapter.py``). pacemaker ignores unknown keys without a word -- here they
are errors.
"""

from __future__ import annotations

import copy
from importlib import resources

import yaml

from ._version import __version__

TOP_KEYS = {"cutoff", "seed", "metadata", "data", "potential", "fit", "backend"}
DATA_KEYS = {"filename", "test_filename", "test_size", "datapath", "reference_energy",
             "ignore_weights", "aug_factor", "seed"}
POTENTIAL_KEYS = {"deltaSplineBins", "elements", "embeddings", "bonds", "functions",
                  "initial_potential", "overwrite_blocks_from_initial_bbasis",
                  "filename", "reset"}
EMBEDDING_KEYS = {"npot", "fs_parameters", "ndensity", "rho_core_cut", "drho_core_cut"}
BOND_KEYS = {"radbase", "radparameters", "rcut", "dcut", "NameOfCutoffFunction",
             "core-repulsion", "r_in", "delta_in", "inner_cutoff_type",
             "nradbase", "nradmax", "lmax"}
FUNCTION_KEYS = {"nradmax_by_orders", "lmax_by_orders", "coefs_init"}
FIT_KEYS = {"loss", "weighting", "optimizer", "maxiter", "options", "fit_cycles",
            "ladder_step", "ladder_type", "noise_relative_sigma", "noise_absolute_sigma",
            "randomize_func_coeffs", "trainable_parameters", "repulsion",
            "early_stopping_patience", "min_relative_train_loss_per_iter",
            "min_relative_test_loss_per_iter", "callbacks"}
LOSS_KEYS = {"kappa", "L1_coeffs", "L2_coeffs", "w0_rad", "w1_rad", "w2_rad", "w_orth"}
ENERGY_WEIGHTING_KEYS = {"type", "nfit", "n_lower", "n_upper", "DElow", "DEup", "DFup",
                         "DE", "DF", "wlow", "energy", "reftype", "seed", "cutoff"}
BACKEND_KEYS = {"evaluator", "batch_size", "batch_size_reduction",
                "batch_size_reduction_factor", "display_step", "gpu_config",
                "parallel_mode", "n_workers", "imagetype"}
BLOCK_ORDER = {"UNARY": 1, "BINARY": 2, "TERNARY": 3, "QUATERNARY": 4, "QUINARY": 5}
HINTS = {
    "L1": "use L1_coeffs", "L2": "use L2_coeffs",
    "gpu": "use backend.gpu_config: {gpu_ind: 0, mem_limit: 0}",
    "func_coefs_init": "use coefs_init",
}


class PacemakerInputError(ValueError):
    pass


def default_template() -> dict:
    text = resources.files("grace2ace").joinpath("templates/pacemaker_default.yaml").read_text()
    return yaml.safe_load(text)


def deep_merge(base: dict, override: dict) -> dict:
    """``override`` wins; dicts merge recursively; ``None`` deletes a key."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def render(cfg: dict, train_file, test_file, reference_energy: dict) -> dict:
    """Template + user overrides + the parts grace2ace owns."""
    from .calculators import teacher_label

    user = cfg["fit"]["input"]
    inp = deep_merge(default_template(), user)
    elements = list(cfg["system"]["elements"])

    pot = inp.setdefault("potential", {})
    # pacemaker lets UNARY/BINARY/... override ALL. So when the user writes an ALL
    # block, the template's more specific blocks the user did NOT write must go,
    # or they would silently overrule the user's ALL (e.g. functions: {ALL: ...}
    # vs the template's UNARY/BINARY/TERNARY). Same-named blocks still merge.
    user_pot = user.get("potential") if isinstance(user.get("potential"), dict) else {}
    if isinstance(pot, dict):
        for section in ("embeddings", "bonds", "functions"):
            user_sec = user_pot.get(section)
            if isinstance(user_sec, dict) and "ALL" in user_sec and isinstance(pot.get(section), dict):
                pot[section] = {k: v for k, v in pot[section].items()
                                if k == "ALL" or k in user_sec or k not in BLOCK_ORDER}
    if isinstance(pot, dict) and "filename" not in pot:
        pot["elements"] = elements
        # e.g. a TERNARY block in a binary run: nothing for it to describe
        funcs = pot.get("functions", {})
        for block in list(funcs):
            if BLOCK_ORDER.get(block, 0) > len(elements):
                funcs.pop(block)

    inp["cutoff"] = max_rcut(inp)
    inp["seed"] = int(cfg["system"]["seed"])
    inp["metadata"] = {
        "name": cfg["system"]["name"],
        "teacher": teacher_label(cfg),
        "made_by": f"grace2ace {__version__}",
    }
    inp["data"] = {
        "filename": str(train_file),
        "test_filename": str(test_file),
        "reference_energy": {k: float(v) for k, v in reference_energy.items()},
    }
    return inp


def max_rcut(inp: dict) -> float:
    pot = inp.get("potential")
    rcuts = []
    if isinstance(pot, dict):
        for block in (pot.get("bonds") or {}).values():
            if isinstance(block, dict) and "rcut" in block:
                rcuts.append(float(block["rcut"]))
    return max(rcuts) if rcuts else float(inp.get("cutoff", 6.0))


def _unknown(keys, allowed, where, problems):
    for key in keys:
        if key not in allowed:
            hint = f" ({HINTS[key]})" if key in HINTS else ""
            problems.append(f"{where}: unknown key '{key}'{hint}")


def check(inp: dict, n_below_delow: int | None = None) -> list[str]:
    """Return a list of problems; empty means the input is safe to run."""
    problems: list[str] = []
    _unknown(inp, TOP_KEYS, "top level", problems)
    for section in ("potential", "fit", "backend", "data"):
        if section not in inp:
            # pacemaker crashes with an unrelated AttributeError without them
            problems.append(f"missing section '{section}'")

    _unknown(inp.get("data", {}), DATA_KEYS, "data", problems)

    pot = inp.get("potential", {})
    if isinstance(pot, dict):
        _unknown(pot, POTENTIAL_KEYS, "potential", problems)
        for name, block in (pot.get("embeddings") or {}).items():
            _unknown(block, EMBEDDING_KEYS, f"potential.embeddings.{name}", problems)
        ndens = {name: b["ndensity"] for name, b in (pot.get("embeddings") or {}).items() if "ndensity" in b}
        if len(set(ndens.values())) > 1:
            problems.append(f"potential.embeddings: tensorpot needs the same ndensity everywhere, got {ndens}")
        for name, block in (pot.get("bonds") or {}).items():
            _unknown(block, BOND_KEYS, f"potential.bonds.{name}", problems)
        if "filename" not in pot and pot.get("bonds") and \
                not any("radparameters" in b for b in pot["bonds"].values()):
            problems.append("potential.bonds: 'radparameters' missing -> pacemaker segfaults "
                            "(the value is ignored for SBessel, but the key must exist)")
        for name, block in (pot.get("functions") or {}).items():
            if name == "number_of_functions_per_element":
                continue
            _unknown(block, FUNCTION_KEYS, f"potential.functions.{name}", problems)

    fit = inp.get("fit", {})
    _unknown(fit, FIT_KEYS, "fit", problems)
    _unknown(fit.get("loss", {}), LOSS_KEYS, "fit.loss", problems)
    weighting = fit.get("weighting") or {}
    if weighting.get("type") == "EnergyBasedWeightingPolicy":
        _unknown(weighting, ENERGY_WEIGHTING_KEYS, "fit.weighting", problems)
        nfit = weighting.get("nfit")
        if nfit is not None and n_below_delow is not None and nfit <= n_below_delow:
            problems.append(
                f"fit.weighting.nfit={nfit} but {n_below_delow} structures sit below "
                f"DElow={weighting.get('DElow', 1.0)}: the whole budget would come from the low "
                f"tier and nothing above DElow would be fitted. Raise nfit above {n_below_delow}."
            )
    _unknown(inp.get("backend", {}), BACKEND_KEYS, "backend", problems)
    return problems


def save(inp: dict, path):
    with open(path, "w") as f:
        f.write(f"# written by grace2ace {__version__}; edit the run config, not this file\n")
        yaml.safe_dump(inp, f, sort_keys=False, default_flow_style=None)
