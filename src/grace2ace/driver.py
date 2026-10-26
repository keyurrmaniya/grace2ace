"""Orchestration: work out what is missing, submit it, report progress.

All decisions are made from the files in the run directory (``*.done``
markers, ``fit/status.json``), never from memory, so ``resume`` is always
safe to call: it only resubmits what has not finished.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import config, io
from .executors import LocalExecutor, SlurmExecutor, _array_spec
from .stages import STAGES


def missing_work(cfg, run_dir) -> dict:
    """stage -> list of missing shards (sharded stages) or bool (others)."""
    n = int(cfg["assyst"]["n_shards"])
    out = {}
    reuse = bool(cfg["dataset"].get("reuse_from"))
    new = io.generates_structures(cfg)
    for stage in ("generate", "label"):
        done = set(io.shards_done(run_dir, stage, n))
        out[stage] = [i for i in range(n) if i not in done] if new else []
    for stage in ("build_df", "fit", "evaluate"):
        out[stage] = not io.is_done(run_dir, stage)
    if reuse and not io.reuse_copy(cfg):   # the dataset comes from another run, as is
        out["build_df"] = False
    return out


def create_run(config_path, force=False, executor=None) -> tuple[dict, Path]:
    cfg = config.load(config_path)
    if executor is not None:
        cfg["executor"] = executor
    run_dir = Path(cfg["system"]["run_dir"])
    frozen = run_dir / "config.yaml"
    if frozen.exists():
        if config.load_frozen(run_dir) != cfg and not force:
            raise SystemExit(
                f"{run_dir} already holds a run with a DIFFERENT config.\n"
                f"Use a new system.name/run_dir, or `grace2ace resume {run_dir}` to continue the old one."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config.freeze(cfg, run_dir)
    shutil.copy(config_path, run_dir / "config.original.yaml")
    if not io.state_path(run_dir).exists():
        io.save_state(run_dir, {"jobs": []})
    return cfg, run_dir


def submit_missing(cfg, run_dir, dry_run=False):
    """Queue everything that is not done yet, chained by dependencies."""
    if not io.reuse_source_ready(cfg):
        msg = (f"dataset.reuse_from: {cfg['dataset']['reuse_from']}/build_df is not finished yet "
               f"(no DONE marker). Submit this run after that one has built its dataset.")
        if not dry_run:
            raise SystemExit(msg)
        print("[dry-run] WARNING:", msg)
    if cfg["executor"] == "local":
        return run_local(cfg, run_dir, dry_run)
    ex = SlurmExecutor(cfg, run_dir, dry_run)
    todo = missing_work(cfg, run_dir)
    if cfg["executor"] == "slurm_single":
        if not any(bool(v) for v in todo.values()):
            print("nothing to do: every stage is done")
            return
        _clear_running_status(run_dir)
        ex.submit("single", attempt=_n_jobs(run_dir, "single"))
        return
    prev = None
    gen_shards = todo["generate"]
    if gen_shards:
        prev = ex.submit("generate", gen_shards)
    if todo["label"]:
        if prev is not None and gen_shards == todo["label"]:
            # task i of label waits only for task i of generate
            prev = ex.submit("label", todo["label"], after=prev, dep_type="aftercorr")
        else:
            prev = ex.submit("label", todo["label"], after=prev)
    if todo["build_df"]:
        prev = ex.submit("build_df", after=prev)
    if todo["fit"]:
        _clear_running_status(run_dir)
        ex.submit("fit", after=prev, attempt=len(_attempt_dirs(run_dir)))
    elif todo["evaluate"]:
        ex.submit("evaluate", after=prev)
    if not any(bool(v) for v in todo.values()):
        print("nothing to do: every stage is done")


def advance(run_dir, dry_run=False):
    """Called at the end of each fit job (or single job): resubmit, or go on to evaluate."""
    cfg = config.load_frozen(run_dir)
    status = io.read_json(io.stage_dir(run_dir, "fit") / "status.json", {}).get("status")
    if cfg["executor"] == "slurm_single":
        return _advance_single(cfg, run_dir, status, dry_run)
    if cfg["executor"] != "slurm":
        return status
    ex = SlurmExecutor(cfg, run_dir, dry_run)
    if status in ("resume", "continue"):
        ex.submit("fit", attempt=len(_attempt_dirs(run_dir)))
    elif status == "done":
        ex.submit("evaluate")
    else:
        print(f"fit status is {status!r}; not submitting anything. See {run_dir}/fit/status.json")
    return status


def _advance_single(cfg, run_dir, status, dry_run):
    """End of a slurm_single job: resubmit it only if the fit stopped for lack of time."""
    if io.is_done(run_dir, "evaluate"):
        print("all stages done")
        return status
    if status not in ("resume", "continue"):
        print(f"not resubmitting: fit status is {status!r} (see {run_dir}/fit/status.json "
              f"and the job log); fix the cause, then `grace2ace resume {run_dir}`")
        return status
    n = _n_jobs(run_dir, "single")
    if n >= int(cfg["fit"]["max_attempts"]):
        print(f"not resubmitting: {n} single jobs already used (fit.max_attempts)")
        return status
    SlurmExecutor(cfg, run_dir, dry_run).submit("single", attempt=n)
    return status


def run_local(cfg, run_dir, dry_run=False, stop_on_resume=False):
    """Every missing stage, in turn, here.

    ``stop_on_resume``: inside one SLURM job (executor slurm_single) a fit that
    stopped at the walltime cannot continue in the same job -- return "resume"
    and let ``advance`` resubmit the job.
    """
    ex = LocalExecutor(cfg, run_dir, dry_run)
    todo = missing_work(cfg, run_dir)
    for stage in ("generate", "label"):
        ex.run_many(stage, todo[stage], int(cfg["assyst"]["parallel_shards"]))
    if todo["build_df"]:
        ex.run("build_df")
    if todo["fit"]:
        _clear_running_status(run_dir)
        while True:
            ex.run("fit")
            if dry_run:
                break
            status = io.read_json(io.stage_dir(run_dir, "fit") / "status.json", {}).get("status")
            if status == "continue":      # more cycles to go, same job
                continue
            if status != "resume" or stop_on_resume:
                break
        if not dry_run and status == "resume" and stop_on_resume:
            print("fit stopped before the walltime; the job will be resubmitted to continue it")
            return "resume"
        if not dry_run and status != "done":
            raise SystemExit(f"fit ended with status {status!r}; see {run_dir}/fit/status.json")
    ex.run("evaluate")
    return "done"


def resume(run_dir, dry_run=False):
    """Cancel this run's queued jobs, then resubmit whatever is missing."""
    cfg = config.load_frozen(run_dir)
    if cfg["executor"] in ("slurm", "slurm_single"):
        ex = SlurmExecutor(cfg, run_dir, dry_run)
        ids = [j["job_id"] for j in io.load_state(run_dir)["jobs"]]
        queued = ex.queue_states(ids)
        if queued:
            print(f"cancelling queued/running jobs of this run: {', '.join(queued)}")
            ex.cancel(list(queued))
    submit_missing(cfg, run_dir, dry_run)


def status(run_dir) -> str:
    cfg = config.load_frozen(run_dir)
    n = int(cfg["assyst"]["n_shards"])
    todo = missing_work(cfg, run_dir)
    lines = [f"run      {cfg['system']['name']}  ({run_dir})"]
    if cfg["dataset"].get("reuse_from"):
        lines.append(f"dataset  reused from {cfg['dataset']['reuse_from']} "
                     f"({'ready' if io.reuse_source_ready(cfg) else 'NOT ready yet'})")
        if io.generates_structures(cfg):
            for stage in ("generate", "label"):
                lines.append(f"{stage:9s}{n - len(todo[stage])}/{n} shards done (seeds/boost)"
                             + (f"  (missing {_array_spec(todo[stage])})" if todo[stage] else ""))
        if io.reuse_copy(cfg):
            lines.append(f"build_df own copy (filters {cfg['dataset'].get('reuse_filters') or 'none'}"
                         f"{' + new structures' if io.generates_structures(cfg) else ''}): "
                         f"{'done' if not todo['build_df'] else '-'}")
    else:
        for stage in ("generate", "label"):
            lines.append(f"{stage:9s}{n - len(todo[stage])}/{n} shards done"
                         + (f"  (missing {_array_spec(todo[stage])})" if todo[stage] else ""))
        lines.append(f"build_df {'done' if not todo['build_df'] else '-'}")
    for stage in ("fit", "evaluate"):
        lines.append(f"{stage:9s}{'done' if not todo[stage] else '-'}")

    summary = io.read_json(io.data_dir(cfg, run_dir) / "summary.json")
    if summary:
        lines.append(f"dataset  {summary['n_train']} train / {summary['n_test']} test structures, "
                     f"{summary['n_families']} families")
    fit_status = io.read_json(io.stage_dir(run_dir, "fit") / "status.json")
    if fit_status:
        lines.append(f"fit      status={fit_status.get('status')} attempt={fit_status.get('attempt', '-')} "
                     f"reason={fit_status.get('reason', '-')}")
    attempts = _attempt_dirs(run_dir)
    if attempts:
        from .stages.fit import last_progress

        prog = last_progress(attempts[-1] / "log.txt")
        if prog:
            lines.append(f"         {attempts[-1].name}: iteration {prog.get('iteration')} "
                         f"loss {prog.get('loss')}")
    report = io.read_json(io.stage_dir(run_dir, "evaluate") / "report.json")
    if report:
        o = report["test"]["overall"]
        lines.append(f"test     E {o['E_rmse_meV_at']:.1f} meV/atom, F {o['F_rmse_meV_A']:.1f} meV/A "
                     f"({o['N']} structures)")
        for b, s in (report["test"].get("by_hull_distance_eV_at") or {}).items():
            if s.get("N"):
                lines.append(f"         hull {b:>9s} eV/at: E {s['E_rmse_meV_at']:6.1f}  "
                             f"F {s['F_rmse_meV_A']:6.1f}  N={s['N']}")

    if cfg["executor"] in ("slurm", "slurm_single"):
        jobs = io.load_state(run_dir)["jobs"]
        queued = SlurmExecutor.queue_states([j["job_id"] for j in jobs])
        for j in jobs[-8:]:
            lines.append(f"job      {j['job_id']:>10s} {j['stage']:9s} {queued.get(j['job_id'], 'not in queue')}")
    return "\n".join(lines)


def _n_jobs(run_dir, stage) -> int:
    return sum(j["stage"] == stage for j in io.load_state(run_dir)["jobs"])


def _attempt_dirs(run_dir):
    return sorted(io.stage_dir(run_dir, "fit").glob("attempt_[0-9][0-9]"))


def _clear_running_status(run_dir):
    """A 'running' status left by a killed job must not block a resubmit."""
    path = io.stage_dir(run_dir, "fit") / "status.json"
    st = io.read_json(path)
    if st and st.get("status") in ("running", "failed"):
        path.unlink()


__all__ = ["create_run", "submit_missing", "advance", "resume", "status", "missing_work", "STAGES"]
