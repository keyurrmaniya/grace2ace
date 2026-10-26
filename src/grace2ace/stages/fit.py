"""Stage 4 (pacemaker env): fit, with automatic resume after a walltime stop.

Every attempt runs in its own folder ``fit/attempt_NN`` so nothing is
overwritten. pacemaker is stopped by us shortly before the SLURM walltime;
the attempt then reports ``resume`` and the next attempt continues from the
latest ``interim_potential_<cycle>.yaml`` with the remaining fit cycles.

The outcome is written to ``fit/status.json``: done | resume | failed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .. import config, io, pacemaker_input

log = logging.getLogger(__name__)


def walltime_seconds(text) -> int:
    """SLURM time ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS' or minutes) -> seconds."""
    text = str(text)
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 1:
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = parts
    return ((days * 24 + h) * 60 + m) * 60 + s


def slurm_time_left():
    """Seconds left in the SLURM job this process runs in, or None outside SLURM."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return walltime_seconds(out.splitlines()[0]) if out else None
    except Exception:  # UNLIMITED, NOT_SET, squeue missing ...
        return None


def fit_time_budget(cfg):
    """How long pacemaker may run before we stop it (None = no limit).

    Inside a SLURM job: the time the job has LEFT minus the margin -- right for
    a per-stage fit job and for a single job that already spent hours on the
    structures. Outside SLURM: fit.local_timeout_hours, if set.
    """
    margin = 60 * float(cfg["fit"]["timeout_margin_minutes"])
    left = slurm_time_left()
    if left is not None:
        return left - margin
    if cfg["executor"] == "slurm":  # squeue unavailable: assume a fresh fit job
        return walltime_seconds(cfg["slurm"]["fit"]["time"]) - margin
    if cfg["fit"]["local_timeout_hours"]:
        return 3600 * float(cfg["fit"]["local_timeout_hours"])
    return None


def tool(name) -> str:
    """A pacemaker CLI from the same env as this python, else from PATH."""
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"'{name}' not found next to {sys.executable} or on PATH")
    return found


def attempts(run_dir) -> list[Path]:
    return sorted(io.stage_dir(run_dir, "fit").glob("attempt_[0-9][0-9]"))


def latest_interim(attempt_dir: Path):
    """(path, cycle) of the newest per-cycle interim potential, or (None, -1)."""
    best, cycle = None, -1
    for p in attempt_dir.glob("interim_potential_[0-9]*.yaml"):
        m = re.fullmatch(r"interim_potential_(\d+)\.yaml", p.name)
        if m and int(m.group(1)) > cycle:
            best, cycle = p, int(m.group(1))
    return best, cycle


_ITERATION = re.compile(r"Iteration:?\s+#(\d+)")


def attempt_progress(log_file: Path) -> tuple[int, int]:
    """(cycles finished, BFGS iterations done in the cycle that was cut off) for one attempt.

    A cycle starts at "Running fit backend" and ends at "Fitting cycle finished";
    pacemaker numbers iterations from 1 in every cycle. Statistics printed after a
    cycle ends also say "Iteration: #N" -- they are not counted.
    """
    finished, open_iters, in_cycle = 0, 0, False
    if not log_file.exists():
        return 0, 0
    with open(log_file, errors="replace") as f:
        for line in f:
            if "Running fit backend" in line:
                in_cycle, open_iters = True, 0
            elif "Fitting cycle finished" in line:
                finished, in_cycle, open_iters = finished + 1, False, 0
            elif in_cycle:
                m = _ITERATION.search(line)
                if m:
                    open_iters = max(open_iters, int(m.group(1)))
    return finished, (open_iters if in_cycle else 0)


def overall_progress(previous: list[Path]) -> tuple[int, int]:
    """(cycle, iterations already done in it) after all previous attempts.

    Each attempt stores in plan.json where it started (cycle, iteration offset);
    its log says how far it got from there.
    """
    cycle, iters = 0, 0
    for prev in previous:
        plan = io.read_json(prev / "plan.json", {"cycle": cycle, "iteration": 0})
        finished, open_iters = attempt_progress(prev / "log.txt")
        if finished:
            cycle, iters = plan["cycle"] + finished, open_iters
        else:
            cycle, iters = plan["cycle"], plan["iteration"] + open_iters
    return cycle, iters


def start_potential(previous: list[Path]):
    """The newest potential left by an earlier attempt (final output beats interim)."""
    for prev in reversed(previous):
        if (prev / "output_potential.yaml").exists():
            return prev / "output_potential.yaml"
        pot, _ = latest_interim(prev)
        if pot is not None:
            return pot
    return None


def wall_start(potential_yaml: Path) -> dict:
    """ZBL wall start r_in per bond, as written in a pacemaker B-basis yaml."""
    out, block = {}, None
    for line in Path(potential_yaml).read_text().splitlines():
        m = re.match(r"\s*- speciesblock:\s*(.+)", line)
        if m:
            block = "-".join(m.group(1).split())
            continue
        m = re.match(r"\s*r_in:\s*([0-9.eE+-]+)", line)
        if m and block and block.count("-") <= 1:
            out[block] = round(float(m.group(1)), 4)
    return out


def apply_pace_corerep(fit_dir: Path, name: str, train_file: Path) -> dict:
    """Replace the `repulsion: auto` wall with the one pace_corerep chooses.

    pace_activeset builds the active set (.asi) on the training data; pace_corerep then
    squeezes small training cells and starts the wall where ACE stops being reliable
    (energy kink or extrapolation grade gamma > 10). If either tool fails, the auto
    wall is kept and the reason is returned (and shown in fit/status.json).
    """
    pot = fit_dir / f"{name}.yaml"
    auto = fit_dir / f"{name}_auto_repulsion.yaml"
    shutil.copy(pot, auto)
    result = {"core_repulsion": "pace_corerep", "r_in_auto": wall_start(auto)}
    with open(fit_dir / "corerep.log", "w") as logf:
        for cmd in ([tool("pace_activeset"), pot.name, "-d", str(train_file)],
                    [tool("pace_corerep"), pot.name, "-d", str(train_file), "-a", f"{name}.asi", "-o", pot.name]):
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=fit_dir, stdout=logf, stderr=subprocess.STDOUT).returncode
            result[f"{Path(cmd[0]).name}_s"] = round(time.time() - t0)
            if rc != 0:
                shutil.copy(auto, pot)
                result.update(core_repulsion="auto",
                              warning=f"{Path(cmd[0]).name} exited with code {rc}; kept the auto wall "
                                      f"(see {fit_dir / 'corerep.log'})")
                log.warning(result["warning"])
                return result
    result["r_in"] = wall_start(pot)
    log.info("ZBL wall start per bond: pace_corerep %s (auto was %s)", result["r_in"], result["r_in_auto"])
    return result


def bad_cells(data_dir) -> list[str]:
    """Cells that would crash pacemaker's neighbour list (io.pacemaker_cell_ok), as problems."""
    problems = []
    for split in ("train", "test"):
        path = Path(data_dir) / f"{split}.pckl.gzip"
        if not path.exists():
            continue
        df = pd.read_pickle(path, compression="gzip")
        bad = df[~df["ase_atoms"].map(lambda a: io.pacemaker_cell_ok(a.cell[:]))]
        if len(bad):
            names = ", ".join(bad["name"].astype(str).head(3)) if "name" in bad else ""
            problems.append(
                f"{len(bad)} {split} cells are so badly shaped that pacemaker's cell inversion fails "
                f"(e.g. {names}). Rebuild the dataset with this grace2ace (it Minkowski-reduces them, labels "
                f"unchanged): delete {Path(data_dir) / 'DONE'} and `grace2ace resume`.")
    return problems


def n_below_delow(data_dir, weighting: dict):
    """How many structures pacemaker will put in the low tier (same filters it applies)."""
    path = Path(data_dir) / "energetics.csv.gz"
    if not path.exists():
        return None
    view = pd.read_csv(path)
    if "hull_distance" not in view or weighting.get("energy", "convex_hull") != "convex_hull":
        return None
    if weighting.get("DFup") is not None:
        view = view[view["fmax"] <= float(weighting["DFup"])]
    view = view[view["hull_distance"] < float(weighting.get("DEup", 10.0))]
    return int((view["hull_distance"] < float(weighting.get("DElow", 1.0))).sum())


def last_progress(log_file: Path) -> dict:
    """Last iteration number and loss from a pacemaker log.txt."""
    out = {}
    if not log_file.exists():
        return out
    pattern = re.compile(r"Iteration:?\s+#(\d+).*?(?:Loss:\s+([0-9.eE+-]+)|Total:\s+([0-9.eE+-]+))")
    with open(log_file, errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                out["iteration"] = int(m.group(1))
                out["loss"] = float(m.group(2) or m.group(3))
            if "Optimization result" in line:
                out["result"] = line.split(" - ", 1)[-1].strip()
    return out


def _set_status(run_dir, status, **extra):
    io.write_json(io.stage_dir(run_dir, "fit") / "status.json", {"status": status, **extra})


def run(cfg: dict, run_dir, shard=None) -> str:
    run_dir = Path(run_dir)
    fit_dir = io.stage_dir(run_dir, "fit")
    if io.is_done(run_dir, "fit"):
        return "done"
    fit_dir.mkdir(parents=True, exist_ok=True)
    source = io.dataset_source(cfg, run_dir)
    if not io.dataset_ready(cfg, run_dir):
        msg = (f"the dataset is not ready: {source}/build_df is not done"
               + (" (dataset.reuse_from) -- submit this run after that one has built its dataset"
                  if source != run_dir else ""))
        _set_status(run_dir, "failed", reason=msg)
        raise RuntimeError(msg)
    if source != run_dir:
        theirs = config.load_frozen(source)["system"]["elements"]
        if sorted(theirs) != sorted(cfg["system"]["elements"]):
            msg = f"dataset.reuse_from {source} has elements {theirs}, this run {cfg['system']['elements']}"
            _set_status(run_dir, "failed", reason=msg)
            raise RuntimeError(msg)
        log.info("fitting on the dataset of %s", source)

    previous = attempts(run_dir)
    k = len(previous)
    max_attempts = int(cfg["fit"]["max_attempts"])
    if k >= max_attempts:
        _set_status(run_dir, "failed", reason=f"{k} attempts used (fit.max_attempts={max_attempts})")
        return "failed"
    timeout = fit_time_budget(cfg)
    if timeout is not None and timeout < 60 * float(cfg["fit"]["min_start_minutes"]):
        # e.g. a single job whose structure generation used up the walltime:
        # do not start (loading the data alone takes minutes), let the next job fit
        _set_status(run_dir, "resume", reason=f"only {timeout / 60:.0f} min of job time left for the fit")
        return "resume"
    attempt_dir = fit_dir / f"attempt_{k:02d}"
    attempt_dir.mkdir()

    data = io.data_dir(cfg, run_dir)
    ref = io.read_json(data / "reference_energy.json")
    inp = pacemaker_input.render(cfg, (data / "train.pckl.gzip").resolve(),
                                 (data / "test.pckl.gzip").resolve(), ref)

    # Where does this attempt start? Continue EXACTLY where the last one stopped:
    # a cycle cut off at iteration d of maxiter M continues with M - d iterations
    # (on its own), then the remaining cycles follow with the full maxiter.
    total_cycles = int(inp["fit"].get("fit_cycles", 1))
    maxiter = int(inp["fit"].get("maxiter", 100))
    cycle, done_iters = overall_progress(previous)
    if done_iters >= maxiter - 5:          # the cut-off cycle was practically complete
        cycle, done_iters = cycle + 1, 0
    if done_iters > 0:
        plan = {"cycle": cycle, "iteration": done_iters, "fit_cycles": 1, "maxiter": maxiter - done_iters}
    elif cycle < total_cycles:
        plan = {"cycle": cycle, "iteration": 0, "fit_cycles": total_cycles - cycle, "maxiter": maxiter}
    else:                                  # all cycles ran but the final files were never written
        plan = {"cycle": total_cycles - 1, "iteration": maxiter, "fit_cycles": 1, "maxiter": 1}
    plan["last"] = plan["cycle"] + plan["fit_cycles"] >= total_cycles
    io.write_json(attempt_dir / "plan.json", plan)
    inp["fit"]["fit_cycles"] = plan["fit_cycles"]
    inp["fit"]["maxiter"] = plan["maxiter"]
    start = start_potential(previous)
    if start is not None:
        local = attempt_dir / "start_potential.yaml"
        shutil.copy(start, local)
        inp["potential"] = local.name
        log.info("attempt %d continues from %s: cycle %d of %d from iteration %d, %d iterations x %d cycle(s)",
                 k, start, plan["cycle"] + 1, total_cycles, plan["iteration"], plan["maxiter"], plan["fit_cycles"])

    weighting = inp["fit"].get("weighting") or {}
    problems = pacemaker_input.check(inp, n_below_delow(data, weighting))
    problems += bad_cells(data)
    if problems:
        # keep the rejected input for inspection, but do not burn an attempt
        pacemaker_input.save(inp, fit_dir / "rejected_input.yaml")
        shutil.rmtree(attempt_dir)
        msg = ("pacemaker input rejected (fix fit.input in the run's config.yaml, then `grace2ace resume`):\n  - "
               + "\n  - ".join(problems))
        _set_status(run_dir, "failed", reason=msg)
        raise ValueError(msg)
    pacemaker_input.save(inp, attempt_dir / "input.yaml")

    _set_status(run_dir, "running", attempt=k)
    t0 = time.time()
    timed_out = False
    with open(attempt_dir / "pacemaker.out", "w") as out:
        # pacemaker, plus runtime fixes for numpy >= 2.4 and a GPU check (see pacemaker_compat)
        proc = subprocess.Popen([sys.executable, "-m", "grace2ace.pacemaker_compat", "input.yaml"], cwd=attempt_dir,
                                stdout=out, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.send_signal(signal.SIGTERM)
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    elapsed = time.time() - t0
    progress = last_progress(attempt_dir / "log.txt")
    name = cfg["system"]["name"]

    final = attempt_dir / "output_potential.yaml"
    if final.exists() and not timed_out and rc == 0 and not plan["last"]:
        # finished the cut-off cycle; the remaining cycles start right away (no new job needed)
        _set_status(run_dir, "continue", attempt=k, elapsed_s=elapsed, **progress)
        return "continue"
    if final.exists() and not timed_out and rc == 0:
        shutil.copy(final, fit_dir / f"{name}.yaml")
        wall = {"core_repulsion": "auto"}
        if cfg["fit"].get("core_repulsion", "auto") == "pace_corerep":
            wall = apply_pace_corerep(fit_dir, name, (data / "train.pckl.gzip").resolve())
        subprocess.run([tool("pace_yaml2yace"), str(fit_dir / f"{name}.yaml"), "-o", str(fit_dir / f"{name}.yace")],
                       check=True, cwd=fit_dir)
        for f in ("test_pred.pckl.gzip", "train_pred.pckl.gzip", "log.txt"):
            if (attempt_dir / f).exists():
                shutil.copy(attempt_dir / f, fit_dir / f)
        info = {"attempts": k + 1, "elapsed_s": elapsed, **progress, "wall": wall}
        _set_status(run_dir, "done", **info)
        io.mark_done(run_dir, "fit", None, info)
        return "done"

    pot, _ = latest_interim(attempt_dir)
    if pot is not None:
        # always leave a usable potential behind, even mid-fit
        subprocess.run([tool("pace_yaml2yace"), str(pot), "-o", str(fit_dir / f"{name}_interim.yace")],
                       check=False, cwd=fit_dir)
    if timed_out and pot is not None and k + 1 < max_attempts:
        _set_status(run_dir, "resume", attempt=k, elapsed_s=elapsed, **progress)
        return "resume"
    reason = "timed out with no attempts left" if timed_out else f"pacemaker exited with code {rc}"
    tail = (attempt_dir / "pacemaker.out").read_text(errors="replace").splitlines()[-30:]
    _set_status(run_dir, "failed", attempt=k, reason=reason, log_tail=tail, **progress)
    return "failed"
