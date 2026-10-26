"""executor: slurm_single -- one job runs every stage and resubmits itself if the fit must stop."""

import yaml
from importlib import resources

from grace2ace import config, driver, io
from grace2ace.executors import SlurmExecutor
from grace2ace.stages import fit


def single_cfg(tmp_path):
    cfg = yaml.safe_load(resources.files("grace2ace").joinpath("examples/NiAlW_00.yaml").read_text())
    cfg["executor"] = "slurm_single"
    cfg["system"]["run_dir"] = str(tmp_path)
    cfg["slurm"]["single"] = {"partition": "fat_gpu", "time": "48:00:00", "gpus": 1, "cpus": 12}
    return config.validate(cfg)


def test_single_job_script(tmp_path):
    cfg = single_cfg(tmp_path)
    text = SlurmExecutor(cfg, tmp_path).script("single", attempt=0).read_text()
    assert "#SBATCH --partition=fat_gpu" in text and "#SBATCH --time=48:00:00" in text
    assert "#SBATCH --gpus=1" in text and "#SBATCH --cpus-per-gpu=12" in text
    assert "--array" not in text
    assert "run-here --run-dir" in text and "advance --run-dir" in text


def test_dry_run_submits_one_job(tmp_path, capsys):
    cfg = single_cfg(tmp_path)
    config.freeze(cfg, tmp_path)
    io.save_state(tmp_path, {"jobs": []})
    driver.submit_missing(cfg, tmp_path, dry_run=True)
    out = capsys.readouterr().out
    assert out.count("sbatch") == 1 and "single_00.sh" in out


def _status(tmp_path, status):
    io.write_json(io.stage_dir(tmp_path, "fit") / "status.json", {"status": status})


def test_advance_resubmits_only_after_a_timed_fit_stop(tmp_path, capsys):
    cfg = single_cfg(tmp_path)
    config.freeze(cfg, tmp_path)
    io.save_state(tmp_path, {"jobs": []})

    _status(tmp_path, "resume")
    driver.advance(tmp_path, dry_run=True)
    assert "single_00.sh" in capsys.readouterr().out

    _status(tmp_path, "failed")
    driver.advance(tmp_path, dry_run=True)
    assert "not resubmitting" in capsys.readouterr().out

    # never more single jobs than fit.max_attempts
    io.save_state(tmp_path, {"jobs": [{"stage": "single", "job_id": str(i)} for i in range(5)]})
    _status(tmp_path, "resume")
    driver.advance(tmp_path, dry_run=True)
    assert "already used" in capsys.readouterr().out


def test_fit_budget_uses_time_left_in_the_job(tmp_path, monkeypatch):
    cfg = single_cfg(tmp_path)
    monkeypatch.setattr(fit, "slurm_time_left", lambda: 30 * 3600)
    assert fit.fit_time_budget(cfg) == 30 * 3600 - 20 * 60
    monkeypatch.setattr(fit, "slurm_time_left", lambda: None)
    cfg["fit"]["local_timeout_hours"] = None
    assert fit.fit_time_budget(cfg) is None


def test_no_fit_start_without_time(tmp_path, monkeypatch):
    cfg = single_cfg(tmp_path)
    io.mark_done(tmp_path, "build_df")
    monkeypatch.setattr(fit, "slurm_time_left", lambda: 25 * 60)  # 25 min left - 20 min margin
    assert fit.run(cfg, tmp_path) == "resume"
    assert not list(io.stage_dir(tmp_path, "fit").glob("attempt_*"))  # no attempt used up


def _log(path, text):
    path.mkdir(parents=True, exist_ok=True)
    (path / "log.txt").write_text(text)


CYCLE = ("I - Running fit backend\n"
         + "".join(f"I - Iteration   #{i}  ({i + 2} evals):      Loss: 1e-3\n" for i in range(1, 501))
         + "I - Optimization result(success=False)\nI - Fitting cycle finished, final statistic:\n"
         + "Iteration:  #501 Loss:    Total:  6.0e-03 (100%)\n")      # stats AFTER the cycle: not counted


def test_progress_counts_iterations_of_the_cut_off_cycle(tmp_path):
    a0 = tmp_path / "attempt_00"
    _log(a0, CYCLE + "I - Running fit backend\n" + "".join(f"I - Iteration   #{i}  (x evals): Loss: 1\n" for i in range(1, 121)))
    assert fit.attempt_progress(a0 / "log.txt") == (1, 120)
    io.write_json(a0 / "plan.json", {"cycle": 0, "iteration": 0})
    assert fit.overall_progress([a0]) == (1, 120)          # cycle index 1, 120 iterations done in it

    # second attempt continued that cycle from iteration 120 and was cut again after 300
    a1 = tmp_path / "attempt_01"
    _log(a1, "I - Running fit backend\n" + "".join(f"I - Iteration   #{i}  (x evals): Loss: 1\n" for i in range(1, 301)))
    io.write_json(a1 / "plan.json", {"cycle": 1, "iteration": 120})
    assert fit.overall_progress([a0, a1]) == (1, 420)


def test_one_long_cycle_is_continued_not_restarted(tmp_path, monkeypatch):
    """3000 iterations in one cycle, cut at 1800 -> the next attempt runs only the last 1200."""
    cfg = single_cfg(tmp_path)
    cfg["fit"]["input"].setdefault("fit", {}).update({"maxiter": 3000, "fit_cycles": 1})
    io.mark_done(tmp_path, "build_df")
    io.write_json(io.stage_dir(tmp_path, "build_df") / "reference_energy.json", {"Ni": 0, "Al": 0, "W": 0, "shift": 0})
    a0 = io.stage_dir(tmp_path, "fit") / "attempt_00"
    _log(a0, "I - Running fit backend\n" + "".join(f"I - Iteration   #{i}  (x evals): Loss: 1\n" for i in range(1, 1801)))
    io.write_json(a0 / "plan.json", {"cycle": 0, "iteration": 0})
    (a0 / "interim_potential_0.yaml").write_text("dummy")

    monkeypatch.setattr(fit, "slurm_time_left", lambda: None)
    monkeypatch.setattr(fit, "n_below_delow", lambda *a: None)
    monkeypatch.setattr(fit.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop here")))
    try:
        fit.run(cfg, tmp_path)
    except RuntimeError:
        pass
    plan = io.read_json(io.stage_dir(tmp_path, "fit") / "attempt_01" / "plan.json")
    assert plan["maxiter"] == 1200 and plan["fit_cycles"] == 1 and plan["last"]
    text = (io.stage_dir(tmp_path, "fit") / "attempt_01" / "input.yaml").read_text()
    assert "maxiter: 1200" in text and "potential: start_potential.yaml" in text
