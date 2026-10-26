"""The teacher: config -> ready-to-use ASE calculator."""

from __future__ import annotations

import os


def get_teacher(cfg: dict):
    """Return the teacher as an ASE calculator (GRACE, or Morse for tests)."""
    teacher = cfg["teacher"]
    params = dict(teacher.get("params") or {})
    if teacher["calculator"] == "grace":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorpotential.calculator import grace_fm

        return grace_fm(teacher["model"], **params)
    if teacher["calculator"] == "morse":
        from ase.calculators.morse import MorsePotential

        return MorsePotential(**params)
    raise ValueError(f"unknown teacher {teacher['calculator']!r}")


def teacher_label(cfg: dict) -> str:
    teacher = cfg["teacher"]
    return teacher["model"] if teacher["calculator"] == "grace" else teacher["calculator"]
