"""Run pacemaker with two small runtime fixes and a GPU check.

``python -m grace2ace.pacemaker_compat input.yaml`` behaves like ``pacemaker input.yaml``, but first:

1. ``FitMetrics.to_FitMetricsDict`` calls ``float()`` on 1-element arrays, which
   numpy >= 2.4 refuses ("only 0-dimensional arrays can be converted"). The
   regularisation terms are turned into true scalars just for that call.
2. ``FitMetrics.compute_metrics`` hides a failure in its "low energy" block with a
   bare ``except: pass``. When that subset is empty, ``low_max_abs_f_comp`` is never
   set and the fit dies one step later with an AttributeError. It now defaults to 0,
   like the other ``low_*`` metrics.
3. It logs which GPUs TensorFlow can see. The old pacemaker env (TF 2.8) found no
   CUDA 11 libraries and fitted on the CPU for weeks without a word.

Nothing installed is modified. Both fixes apply to GRACE's copy of the ACE backend
(``tensorpotential.compat.pace``) and to the old stand-alone TensorPotential.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np


def _fitmetrics_class():
    try:
        from tensorpotential.compat.pace.fitmetrics import FitMetrics
    except ImportError:
        try:
            from tensorpotential.fitmetrics import FitMetrics
        except ImportError:
            return None
    return FitMetrics


def _scalar(x):
    a = np.asarray(x)
    return a.reshape(-1)[0] if a.size == 1 else x


def patch():
    FitMetrics = _fitmetrics_class()
    if FitMetrics is None or getattr(FitMetrics, "_grace2ace_patched", False):
        return
    compute_metrics = FitMetrics.compute_metrics
    to_dict = FitMetrics.to_FitMetricsDict

    def compute_metrics_safe(self, *args, **kwargs):
        self.low_max_abs_f_comp = 0
        return compute_metrics(self, *args, **kwargs)

    def to_dict_safe(self):
        regs, weights = self.regs, self.reg_weights
        self.regs = [_scalar(r) for r in regs]
        self.reg_weights = [_scalar(w) for w in weights]
        try:
            return to_dict(self)
        finally:
            self.regs, self.reg_weights = regs, weights

    FitMetrics.compute_metrics = compute_metrics_safe
    FitMetrics.to_FitMetricsDict = to_dict_safe
    FitMetrics._grace2ace_patched = True


def report_gpus():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
    except Exception as ex:  # report, never block the fit
        print(f"[grace2ace] could not query TensorFlow GPUs: {ex}", flush=True)
        return
    print(f"[grace2ace] TensorFlow {tf.__version__} sees GPUs: {[g.name for g in gpus]}", flush=True)
    if not gpus and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        print("[grace2ace] WARNING: a GPU was allocated (CUDA_VISIBLE_DEVICES="
              f"{os.environ['CUDA_VISIBLE_DEVICES']}) but TensorFlow cannot use it: "
              "this fit runs on the CPU.", flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    script = Path(sys.executable).parent / "pacemaker"
    if not script.exists():
        raise SystemExit(f"pacemaker not found next to {sys.executable}")
    report_gpus()
    patch()
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
