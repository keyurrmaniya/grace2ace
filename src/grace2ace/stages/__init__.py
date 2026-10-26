"""Pipeline stages. Each module exposes ``run(run_dir, shard=None)``."""

STAGES = ("generate", "label", "build_df", "fit", "evaluate")
SHARDED = ("generate", "label")
