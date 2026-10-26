"""Command line: ``grace2ace run|status|resume|init-config|stage|advance|run-here``."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from importlib import resources
from pathlib import Path

from ._version import __version__

EXAMPLES = ("NiAlW_00", "NiAlW_smoke", "local_morse_test")


def _cmd_init_config(args):
    text = resources.files("grace2ace").joinpath(f"examples/{args.example}.yaml").read_text()
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}; edit it, then `grace2ace run {args.output}`")
    else:
        sys.stdout.write(text)


def _cmd_run(args):
    from . import driver

    cfg, run_dir = driver.create_run(args.config, force=args.force,
                                     executor="local" if args.local else None)
    print(f"run directory: {run_dir}")
    driver.submit_missing(cfg, run_dir, dry_run=args.dry_run)


def _cmd_resume(args):
    from . import driver

    driver.resume(Path(args.run_dir).resolve(), dry_run=args.dry_run)


def _cmd_status(args):
    from . import driver

    print(driver.status(Path(args.run_dir).resolve()))


def _cmd_stage(args):
    from . import config
    from .stages import SHARDED

    # ASSYST's per-formula progress bars would fill the SLURM logs
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_dir = Path(args.run_dir).resolve()
    cfg = config.load_frozen(run_dir)
    if args.name in SHARDED and args.shard is None:
        raise SystemExit(f"stage {args.name} needs --shard")
    module = importlib.import_module(f"grace2ace.stages.{args.name}")
    result = module.run(cfg, run_dir, args.shard)
    if args.name == "fit":
        print(f"fit stage result: {result}")


def _cmd_advance(args):
    from . import driver

    driver.advance(Path(args.run_dir).resolve(), dry_run=args.dry_run)


def _cmd_run_here(args):
    from . import config, driver

    run_dir = Path(args.run_dir).resolve()
    result = driver.run_local(config.load_frozen(run_dir), run_dir, stop_on_resume=True)
    print(f"run-here result: {result}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="grace2ace",
        description="Distil a GRACE model into an ACE potential: ASSYST structures -> GRACE labels -> pacemaker.",
    )
    p.add_argument("--version", action="version", version=f"grace2ace {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-config", help="print or write an example run config")
    s.add_argument("example", nargs="?", default="NiAlW_00", choices=EXAMPLES)
    s.add_argument("-o", "--output", help="write to this file instead of stdout")
    s.set_defaults(func=_cmd_init_config)

    s = sub.add_parser("run", help="create the run directory and submit every stage")
    s.add_argument("config")
    s.add_argument("--local", action="store_true", help="run all stages here instead of on SLURM")
    s.add_argument("--dry-run", action="store_true", help="write job scripts but submit nothing")
    s.add_argument("--force", action="store_true", help="reuse a run dir whose config differs")
    s.set_defaults(func=_cmd_run)

    s = sub.add_parser("resume", help="cancel this run's queued jobs and resubmit what is not done")
    s.add_argument("run_dir")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=_cmd_resume)

    s = sub.add_parser("status", help="progress, fit loss and test errors of a run")
    s.add_argument("run_dir")
    s.set_defaults(func=_cmd_status)

    s = sub.add_parser("stage", help="(used by job scripts) run one stage in this python")
    s.add_argument("name", choices=["generate", "label", "build_df", "fit", "evaluate"])
    s.add_argument("--run-dir", required=True)
    s.add_argument("--shard", type=int, default=None)
    s.set_defaults(func=_cmd_stage)

    s = sub.add_parser("advance", help="(used by fit / single jobs) resubmit, or start evaluate")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=_cmd_advance)

    s = sub.add_parser("run-here", help="(used by the slurm_single job) run every missing stage here, in turn")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=_cmd_run_here)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
