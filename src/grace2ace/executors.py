"""Where stages run: SLURM (sbatch with dependencies) or locally (subprocess)."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from importlib import resources
from pathlib import Path

from . import io
from ._version import __version__
from .stages import SHARDED

ENV_OF_STAGE = {"generate": "generate", "label": "generate",
                "build_df": "fit", "fit": "fit", "evaluate": "fit", "single": "fit"}


def stage_command(cfg, run_dir, stage, shard=None) -> list[str]:
    python = cfg["envs"][ENV_OF_STAGE[stage]]
    cmd = [python, "-m", "grace2ace", "stage", stage, "--run-dir", str(run_dir)]
    if shard is not None:
        cmd += ["--shard", str(shard)]
    return cmd


def activation_lines(python) -> list[str]:
    """Shell lines that enter the conda env owning ``python`` (no-op otherwise)."""
    prefix = Path(python).parent.parent
    lines = [f"export PATH={shlex.quote(str(prefix / 'bin'))}:\"$PATH\""]
    if (prefix / "conda-meta").is_dir():
        lines = [
            f"export CONDA_PREFIX={shlex.quote(str(prefix))}",
            lines[0],
            "set +e",
            'for hook in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [ -f "$hook" ] && . "$hook"; done',
            "set -e",
        ]
    return lines


def _array_spec(shards) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8'."""
    shards = sorted(shards)
    parts, start, prev = [], shards[0], shards[0]
    for s in shards[1:] + [None]:
        if s is not None and s == prev + 1:
            prev = s
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        if s is not None:
            start = prev = s
    return ",".join(parts)


class SlurmExecutor:
    def __init__(self, cfg, run_dir, dry_run=False):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.dry_run = dry_run

    def script(self, stage, shards=None, attempt=None) -> Path:
        s = self.cfg["slurm"]
        st = s[stage]
        name = self.cfg["system"]["name"]
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        cpus, gpus = int(st["cpus"]), int(st.get("gpus") or 0)
        lines = ["#SBATCH -N 1", "#SBATCH --ntasks=1",
                 f"#SBATCH --time={st['time']}", "#SBATCH --kill-on-invalid-dep=yes"]
        if gpus > 0:
            # some clusters refuse --cpus-per-task together with GPUs
            lines += [f"#SBATCH --gpus={gpus}", f"#SBATCH --cpus-per-gpu={max(1, cpus // gpus)}"]
        else:
            lines.append(f"#SBATCH --cpus-per-task={cpus}")
        if st.get("partition"):
            lines.append(f"#SBATCH --partition={st['partition']}")
        if s.get("account"):
            lines.append(f"#SBATCH --account={s['account']}")
        if st.get("mem"):
            lines.append(f"#SBATCH --mem={st['mem']}")
        if shards is not None:
            lines.append(f"#SBATCH --array={_array_spec(shards)}")
        lines += [f"#SBATCH {x}" for x in st.get("extra") or []]

        if stage in SHARDED:
            cmd = shlex.join(stage_command(self.cfg, self.run_dir, stage)) + ' --shard "$SLURM_ARRAY_TASK_ID"'
            log = log_dir / f"{stage}_%A_%a.log"
        elif stage == "fit":
            py = self.cfg["envs"]["fit"]
            cmd = (shlex.join(stage_command(self.cfg, self.run_dir, "fit")) + " || true\n"
                   + shlex.join([py, "-m", "grace2ace", "advance", "--run-dir", str(self.run_dir)]))
            log = log_dir / f"fit_{attempt or 0:02d}_%j.log"
        elif stage == "single":
            # every stage in turn on this one allocation; then resubmit if the fit had to stop
            py = self.cfg["envs"]["fit"]
            cmd = (shlex.join([py, "-m", "grace2ace", "run-here", "--run-dir", str(self.run_dir)]) + " || true\n"
                   + shlex.join([py, "-m", "grace2ace", "advance", "--run-dir", str(self.run_dir)]))
            log = log_dir / f"single_{attempt or 0:02d}_%j.log"
        else:
            cmd = shlex.join(stage_command(self.cfg, self.run_dir, stage))
            log = log_dir / f"{stage}_%j.log"

        text = resources.files("grace2ace").joinpath("templates/slurm_job.sh").read_text()
        for key, value in {
            "JOB_NAME": f"{name}_{stage}", "LOG": str(log), "SBATCH_LINES": "\n".join(lines),
            "VERSION": __version__, "SETUP": "\n".join(s.get("setup") or []),
            "ACTIVATE": "\n".join(activation_lines(self.cfg["envs"][ENV_OF_STAGE[stage]])),
            "CPUS": str(int(st["cpus"])), "STAGE": stage, "COMMAND": cmd,
        }.items():
            text = text.replace(f"@@{key}@@", value)
        path = self.run_dir / "jobs" / (f"{stage}.sh" if attempt is None else f"{stage}_{attempt:02d}.sh")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def submit(self, stage, shards=None, after=None, dep_type="afterok", attempt=None):
        """Submit one stage; returns the job id (or a fake one in dry-run)."""
        path = self.script(stage, shards, attempt)
        cmd = ["sbatch", "--parsable"]
        if after:
            cmd.append(f"--dependency={dep_type}:{after}")
        cmd.append(str(path))
        if self.dry_run:
            print("[dry-run]", " ".join(cmd))
            return f"dry-{stage}"
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"sbatch refused {path.name}:\n{res.stderr.strip()}")
        job_id = res.stdout.strip().split(";")[0]
        state = io.load_state(self.run_dir)
        state["jobs"].append({"stage": stage, "job_id": job_id, "shards": shards, "attempt": attempt,
                              "dependency": f"{dep_type}:{after}" if after else None,
                              "submitted": time.strftime("%Y-%m-%d %H:%M:%S")})
        io.save_state(self.run_dir, state)
        print(f"submitted {stage:9s} job {job_id}" + (f" (shards {_array_spec(shards)})" if shards else "")
              + (f" after {after}" if after else ""))
        return job_id

    @staticmethod
    def queue_states(job_ids) -> dict:
        """job id -> SLURM state for jobs still in the queue."""
        if not job_ids:
            return {}
        res = subprocess.run(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i %T %r"],
                             capture_output=True, text=True)
        states = {}
        for line in res.stdout.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2:
                base = parts[0].split("_")[0]
                states.setdefault(base, []).append(" ".join(parts[1:]))
        return {k: ", ".join(sorted(set(v))) for k, v in states.items()}

    def cancel(self, job_ids):
        if job_ids and not self.dry_run:
            subprocess.run(["scancel", *job_ids], check=False)


class LocalExecutor:
    """Runs every stage here, one after another (tests, laptops, interactive nodes)."""

    def __init__(self, cfg, run_dir, dry_run=False):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.dry_run = dry_run

    def run(self, stage, shard=None):
        cmd = stage_command(self.cfg, self.run_dir, stage, shard)
        print("[local]", shlex.join(cmd), flush=True)
        if not self.dry_run:
            subprocess.run(cmd, check=True)

    def run_many(self, stage, shards, n_parallel=1):
        """Run shards with at most ``n_parallel`` at the same time.

        Several GRACE processes then share the one GPU: each may only take the GPU
        memory it needs (TF_FORCE_GPU_ALLOW_GROWTH, as GRACE's own multi-walker tools
        do) and gets an equal share of the CPU cores. Each shard writes to
        logs/<stage>_<shard>.log. A failed shard does not stop the others; the
        failures are reported once all have ended.
        """
        shards = list(shards)
        if n_parallel <= 1 or len(shards) <= 1:
            for shard in shards:
                self.run(stage, shard)
            return
        n_parallel = min(n_parallel, len(shards))
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = os.cpu_count() or 1
        threads = str(max(1, cores // n_parallel))
        env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", OMP_NUM_THREADS=threads,
                   TF_NUM_INTRAOP_THREADS=threads, TF_NUM_INTEROP_THREADS="2")
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[local] {stage}: {len(shards)} shards, {n_parallel} at a time, {threads} CPU threads each",
              flush=True)
        if self.dry_run:
            for shard in shards:
                print("[local]", shlex.join(stage_command(self.cfg, self.run_dir, stage, shard)))
            return

        waiting, running, failed = list(shards), {}, []
        t0 = time.time()
        while waiting or running:
            while waiting and len(running) < n_parallel:
                shard = waiting.pop(0)
                log = open(log_dir / f"{stage}_{shard:03d}.log", "a")
                proc = subprocess.Popen(stage_command(self.cfg, self.run_dir, stage, shard),
                                        stdout=log, stderr=subprocess.STDOUT, env=env)
                running[shard] = (proc, log)
                print(f"[local] {stage} shard {shard} started", flush=True)
            for shard, (proc, log) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del running[shard]
                print(f"[local] {stage} shard {shard} {'done' if rc == 0 else f'FAILED (exit {rc})'} "
                      f"after {(time.time() - t0) / 60:.1f} min", flush=True)
                if rc != 0:
                    failed.append(shard)
            time.sleep(2)
        if failed:
            raise RuntimeError(f"{stage} shards {sorted(failed)} failed; see {log_dir}/{stage}_NNN.log")
