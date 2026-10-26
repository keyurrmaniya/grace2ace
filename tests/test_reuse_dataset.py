"""dataset.reuse_from: fit on another run's finished dataset."""

import pytest
import yaml
from importlib import resources

from grace2ace import config, driver, io
from grace2ace.stages import fit


def cfg_for(run_dir, source=None, elements=("Ni", "Al", "W")):
    raw = yaml.safe_load(resources.files("grace2ace").joinpath("examples/NiAlW_00.yaml").read_text())
    raw["system"]["run_dir"] = str(run_dir)
    raw["system"]["elements"] = list(elements)
    raw["assyst"]["distance_radii"] = {e: 1.0 for e in elements}
    raw["executor"] = "slurm_single"
    if source is not None:
        raw["dataset"]["reuse_from"] = str(source)
    return config.validate(raw)


def test_reuse_skips_the_data_stages(tmp_path):
    cfg = cfg_for(tmp_path / "b", source=tmp_path / "a")
    todo = driver.missing_work(cfg, tmp_path / "b")
    assert todo["generate"] == [] and todo["label"] == [] and todo["build_df"] is False
    assert todo["fit"] and todo["evaluate"]
    assert io.data_dir(cfg, tmp_path / "b") == tmp_path / "a" / "build_df"


def test_no_reuse_means_own_dataset(tmp_path):
    cfg = cfg_for(tmp_path)
    assert io.data_dir(cfg, tmp_path) == tmp_path / "build_df"
    assert driver.missing_work(cfg, tmp_path)["build_df"] is True


def test_refuses_to_submit_before_the_dataset_exists(tmp_path, capsys):
    b = tmp_path / "b"
    cfg = cfg_for(b, source=tmp_path / "a")
    config.freeze(cfg, b.mkdir() or b)
    io.save_state(b, {"jobs": []})
    with pytest.raises(SystemExit, match="not finished yet"):
        driver.submit_missing(cfg, b)
    driver.submit_missing(cfg, b, dry_run=True)            # dry run only warns
    assert "WARNING" in capsys.readouterr().out
    io.mark_done(tmp_path / "a", "build_df")               # once the dataset is there ...
    driver.submit_missing(cfg, b, dry_run=True)            # ... it would submit
    assert "single_00.sh" in capsys.readouterr().out


def test_fit_refuses_a_dataset_with_other_elements(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    config.freeze(cfg_for(a, elements=("Cu", "Ag", "Pd")), a.mkdir() or a)
    io.mark_done(a, "build_df")
    with pytest.raises(RuntimeError, match="elements"):
        fit.run(cfg_for(b, source=a), b)
    assert io.read_json(io.stage_dir(b, "fit") / "status.json")["status"] == "failed"


def test_reuse_from_own_run_dir_is_an_error(tmp_path):
    with pytest.raises(config.ConfigError, match="own run_dir"):
        cfg_for(tmp_path, source=tmp_path)
