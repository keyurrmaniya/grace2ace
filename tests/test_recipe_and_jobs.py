import numpy as np
import pytest
import yaml
from importlib import resources

from grace2ace import config
from grace2ace.executors import SlurmExecutor


def example(name):
    return config.validate(yaml.safe_load(
        resources.files("grace2ace").joinpath(f"examples/{name}.yaml").read_text()))


def test_perturbations_map_onto_assyst():
    pytest.importorskip("assyst")
    from assyst.perturbations import Rattle, Series, Stretch
    from grace2ace import recipe

    cfg = example("NiAlW_00")
    ps = recipe.build_perturbations(cfg["assyst"]["perturbations"], np.random.default_rng(0))
    assert len(ps) == 10
    assert isinstance(ps[0], Stretch) and ps[0].hydro == 0.15
    assert isinstance(ps[5], Rattle) and ps[5].create_supercells
    assert isinstance(ps[-1], Series)


def test_repeat_and_random_choice():
    pytest.importorskip("assyst")
    from assyst.perturbations import RandomChoice
    from grace2ace import recipe

    specs = [{"repeat": 3, "item": {"random_choice": {
        "a": {"rattle": {"sigma": 0.25}}, "b": {"stretch": {"hydro": 0.4, "shear": 0.05}}, "chance": 0.3}}}]
    ps = recipe.build_perturbations(specs, np.random.default_rng(0))
    assert len(ps) == 3 and all(isinstance(p, RandomChoice) for p in ps)
    with pytest.raises(ValueError, match="unknown perturbation"):
        recipe.build_perturbations([{"shake": {}}], np.random.default_rng(0))


def test_formula_count_matches_old_scripts():
    pytest.importorskip("assyst")
    from grace2ace import recipe

    # NiAlW-Ternary/test_05 printed "Total unique formulas to sample: 273"
    assert len(recipe.formulas(example("NiAlW_00"))) == 273
    shards = [recipe.shard_formulas(recipe.formulas(example("NiAlW_00")), i, 10) for i in range(10)]
    assert sum(len(s) for s in shards) == 273


def test_job_script(tmp_path):
    cfg = example("NiAlW_00")
    cfg["slurm"]["account"] = "acct_1"
    ex = SlurmExecutor(cfg, tmp_path)
    text = ex.script("generate", shards=list(range(10))).read_text()
    assert "#SBATCH --array=0-9" in text
    assert "#SBATCH --account=acct_1" in text
    assert "#SBATCH --gpus=1" in text and "#SBATCH --cpus-per-gpu=12" in text
    assert "--cpus-per-task" not in text
    assert "export OMP_NUM_THREADS=12" in text
    assert not [l for l in text.splitlines() if l.strip().startswith(("conda", "source"))]
    assert 'stage generate --run-dir' in text and '--shard "$SLURM_ARRAY_TASK_ID"' in text
    fit = ex.script("fit", attempt=0).read_text()
    assert "advance --run-dir" in fit


@pytest.mark.parametrize("drop", [False, True])
def test_unconverged_relaxations_are_kept_unless_asked(tmp_path, drop):
    """ASSYST >= 0.16 raises NotConvergedError at max_steps; test_05's ASSYST kept those cells."""
    pytest.importorskip("assyst")
    from grace2ace import io
    from grace2ace.stages import generate

    cfg = example("local_morse_test")
    cfg["system"]["run_dir"] = str(tmp_path)
    cfg["assyst"].update({"range": [0, 3], "trim": [2, 3], "structures_per_formula": 3, "n_shards": 1,
                          "drop_unconverged": drop,
                          "relax": [{"type": "VolumeRelax", "max_steps": 2, "force_tolerance": 1e-8}]})
    generate.run(cfg, tmp_path, 0)
    info = io.read_json(io.done_marker(tmp_path, "generate", 0))
    n_unconv = info["relax_unconverged"]["VolumeRelax"]
    assert n_unconv == info["sampled"] > 0 and info["relax_failed"] == {}
    written = io.read_extxyz(tmp_path / "generate" / "shard_000.extxyz")
    assert len(written) == (0 if drop else n_unconv)
    assert all(a.info["relax_converged"] is False for a in written)
