"""assyst.parallel_shards: several shards at once on one node (executor local / slurm_single)."""

import sys
import time

import pytest

from grace2ace import executors
from grace2ace.executors import LocalExecutor


def fake_stage(script):
    """Replace the real stage command by a tiny python program."""
    return lambda cfg, run_dir, stage, shard=None: [sys.executable, "-c", script.format(shard=shard)]


def test_shards_run_at_the_same_time(tmp_path, monkeypatch):
    monkeypatch.setattr(executors, "stage_command", fake_stage("import time; time.sleep(1.5)"))
    t0 = time.time()
    LocalExecutor({}, tmp_path).run_many("generate", range(4), n_parallel=4)
    assert time.time() - t0 < 4.5          # one after another would take >= 6 s
    assert sorted(p.name for p in (tmp_path / "logs").iterdir()) == [f"generate_{i:03d}.log" for i in range(4)]


def test_failed_shard_does_not_stop_the_others(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    marker.mkdir()
    script = ("import sys, pathlib; pathlib.Path(r'%s', str({shard})).touch(); "
              "sys.exit(1 if {shard} == 1 else 0)") % marker
    monkeypatch.setattr(executors, "stage_command", fake_stage(script))
    with pytest.raises(RuntimeError, match=r"shards \[1\] failed"):
        LocalExecutor({}, tmp_path).run_many("label", range(4), n_parallel=2)
    assert sorted(p.name for p in marker.iterdir()) == ["0", "1", "2", "3"]


def test_gpu_memory_growth_and_thread_share(tmp_path, monkeypatch):
    out = tmp_path / "env.txt"
    script = ("import os; open(r'%s', 'a').write(os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] + ' ' "
              "+ os.environ['OMP_NUM_THREADS'] + chr(10))") % out
    monkeypatch.setattr(executors, "stage_command", fake_stage(script))
    monkeypatch.setattr(executors.os, "sched_getaffinity", lambda pid: set(range(16)))
    LocalExecutor({}, tmp_path).run_many("generate", range(4), n_parallel=4)
    assert out.read_text().split("\n")[:4] == ["true 4"] * 4      # 16 cores / 4 shards
