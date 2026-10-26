from grace2ace import config, pacemaker_input as pi


def cfg_with(fit_input=None, elements=("Ni", "Al", "W")):
    return config.validate({
        "system": {"name": "t", "elements": list(elements)},
        "assyst": {"distance_radii": {e: 1.0 for e in elements}},
        "envs": {"generate": "/x", "fit": "/y"},
        "teacher": {"calculator": "morse"},
        "fit": {"input": fit_input or {}},
    })


REF = {"Ni": -1.0, "Al": -2.0, "W": -3.0, "shift": 0.1}


def test_template_passes_its_own_check():
    inp = pi.render(cfg_with(), "/train.pckl.gzip", "/test.pckl.gzip", REF)
    assert pi.check(inp) == []
    assert inp["cutoff"] == 6.0
    assert inp["potential"]["elements"] == ["Ni", "Al", "W"]
    assert inp["data"]["test_filename"] == "/test.pckl.gzip"
    assert inp["fit"]["loss"]["L1_coeffs"] == 1e-8


def test_silently_ignored_keys_are_caught():
    inp = pi.render(cfg_with({"fit": {"loss": {"L1": 1e-6, "L2": 1e-6}}, "backend": {"gpu": 0}}),
                    "/a", "/b", REF)
    problems = "\n".join(pi.check(inp))
    assert "fit.loss: unknown key 'L1' (use L1_coeffs)" in problems
    assert "backend: unknown key 'gpu'" in problems


def test_none_deletes_and_radparameters_required():
    inp = pi.render(cfg_with({"potential": {"bonds": {"ALL": {"radparameters": None}}}}), "/a", "/b", REF)
    assert any("radparameters" in p for p in pi.check(inp))


def test_nfit_below_low_tier():
    inp = pi.render(cfg_with(), "/a", "/b", REF)
    assert any("Raise nfit" in p for p in pi.check(inp, n_below_delow=50000))
    assert pi.check(inp, n_below_delow=30000) == []


def test_mixed_ndensity_rejected():
    inp = pi.render(cfg_with({"potential": {"embeddings": {"Ni": {"ndensity": 3}}}}), "/a", "/b", REF)
    assert any("ndensity" in p for p in pi.check(inp))


def test_higher_order_blocks_dropped_for_binary():
    inp = pi.render(cfg_with(elements=("Ni", "Al")), "/a", "/b", REF)
    assert "TERNARY" not in inp["potential"]["functions"]
    assert set(inp["potential"]["functions"]) == {"UNARY", "BINARY"}


def test_user_functions_replace_the_template_blocks():
    # an ALL block must not be overruled by the template's UNARY/BINARY/TERNARY
    funcs = {"number_of_functions_per_element": 1000,
             "ALL": {"nradmax_by_orders": [15, 6, 4, 3, 2], "lmax_by_orders": [0, 4, 3, 2, 1]}}
    inp = pi.render(cfg_with({"potential": {"functions": funcs}}), "/a", "/b", REF)
    assert inp["potential"]["functions"] == funcs
    assert pi.check(inp) == []
    # not given -> template blocks stay
    assert "UNARY" in pi.render(cfg_with(), "/a", "/b", REF)["potential"]["functions"]
