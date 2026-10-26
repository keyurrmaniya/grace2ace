import pytest
import yaml
from importlib import resources

from grace2ace import config


def minimal(**over):
    cfg = {
        "system": {"name": "t", "elements": ["Ni", "Al"]},
        "assyst": {"distance_radii": {"Ni": 1.0, "Al": 1.1}},
        "envs": {"generate": "/usr/bin/python3", "fit": "/usr/bin/python3"},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_defaults_filled():
    cfg = config.validate(minimal())
    assert cfg["filters"]["max_force"] == 20.0
    assert cfg["assyst"]["relax"][0]["type"] == "VolumeRelax"
    assert cfg["fit"]["input"] == {}
    assert cfg["slurm"]["fit"]["extra"] == []
    assert cfg["system"]["run_dir"].endswith("runs/t")


def test_unknown_key_is_an_error():
    with pytest.raises(config.ConfigError, match="unknown config key 'max_forces'"):
        config.validate(minimal(filters={"max_forces": 1}))


def test_missing_radius():
    cfg = minimal()
    cfg["assyst"]["distance_radii"] = {"Ni": 1.0}
    with pytest.raises(config.ConfigError, match="Al"):
        config.validate(cfg)


def test_missing_required():
    cfg = minimal()
    del cfg["envs"]
    with pytest.raises(config.ConfigError, match="envs.generate"):
        config.validate(cfg)


def test_fit_input_is_free_form():
    cfg = config.validate(minimal(fit={"input": {"fit": {"loss": {"kappa": 0.3}}}}))
    assert cfg["fit"]["input"]["fit"]["loss"]["kappa"] == 0.3


@pytest.mark.parametrize("name", ["NiAlW_00", "NiAlW_smoke", "local_morse_test"])
def test_shipped_examples_validate(name):
    text = resources.files("grace2ace").joinpath(f"examples/{name}.yaml").read_text()
    cfg = config.validate(yaml.safe_load(text))
    assert cfg["system"]["name"] == name


def test_freeze_roundtrip(tmp_path):
    cfg = config.validate(minimal())
    config.freeze(cfg, tmp_path)
    assert config.load_frozen(tmp_path) == cfg


def test_section_given_as_scalar_is_a_clear_error():
    with pytest.raises(config.ConfigError, match="slurm.generate must be a mapping"):
        config.validate(minimal(slurm={"generate": "/some/python"}))


def test_old_frozen_config_gets_new_defaults(tmp_path):
    cfg = config.validate(minimal())
    del cfg["assyst"]["drop_unconverged"], cfg["assyst"]["parallel_shards"]   # as written by an older grace2ace
    config.freeze(cfg, tmp_path)
    again = config.load_frozen(tmp_path)
    assert again["assyst"]["drop_unconverged"] is False and again["assyst"]["parallel_shards"] == 1
    assert again["system"]["run_dir"] == cfg["system"]["run_dir"]
