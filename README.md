# grace2ace

Distil a GRACE foundation model into a fast ACE potential, with one config file and one command.

```
ASSYST random crystals ──► relax with GRACE ──► perturb + GRACE labels ──► pacemaker DataFrame ──► ACE fit ──► test report
      (generate)              (generate)              (label)                    (build_df)            (fit)      (evaluate)
```

GRACE is the **teacher**: accurate, but slow in MD. ACE is the **student**: fast, and trained to copy the teacher's answers. grace2ace builds the student's training set with [ASSYST](https://github.com/eisenforschung/assyst), has GRACE give the answers, and fits ACE with [pacemaker](https://github.com/ICAMS/python-ace). On a SLURM cluster it does this unattended:
- independent shards of the work run in parallel on several GPUs;
- every stage waits for the one before it;
- a fit that hits the walltime continues by itself;
- the result is checked on a test set that shares no structure families with the training set.

## One environment for everything

All five stages run in a single conda env:

| stage | does |
|---|---|
| `generate` | random symmetric crystals (ASSYST/pyxtal), volume + full relaxation with the teacher |
| `label` | strain/rattle perturbations, teacher energies/forces/stress, filters, drop report |
| `build_df` | pacemaker DataFrame, frozen reference energy, test split by structure family |
| `fit` | input check, `pacemaker` on the GPU, auto-resume, `.yace` |
| `evaluate` | errors of the final `.yace` on the whole test set, by distance above the convex hull and by subsystem |

**Why this needs care.** pacemaker's releases fit with the stand-alone `TensorPotential` package. GRACE is also called `tensorpotential`, so the two overwrite each other. The old TensorPotential also needs TensorFlow 2.8 + CUDA 11, and many clusters only provide CUDA 12. Such a fit then **silently runs on the CPU**.

**What we do instead.** GRACE (`grace-tensorpotential` ≥ 0.5.9) ships pacemaker's ACE fitting backend as `tensorpotential.compat.pace`, and pacemaker's `feature/grace_fs` branch uses it. One env then holds everything, on a current TensorFlow whose CUDA libraries come from pip.

Tested on a Ni-Al-W fit (124k structures, 5079 functions): the loss trajectory is identical to the old two-env fit to all printed digits, and each iteration is 3.4× faster (L40S GPU vs 16 CPU cores).

`grace2ace` runs pacemaker through `python -m grace2ace.pacemaker_compat`, which adds three things:
- **numpy ≥ 2.4 fix.** One `float()` call in the backend is made numpy-2.4-safe at run time, so pacemaker's own `numpy<2.2` pin is not needed.
- **Empty-subset fix.** A crash on an empty "low energy" subset is guarded.
- **GPU check.** It logs which GPUs TensorFlow sees, and warns if a GPU was allocated but cannot be used.

Two separate envs (GRACE side and pacemaker side) still work: set `envs.generate` and `envs.fit` to different pythons. Stages hand structures over as extxyz, never as pickles, because numpy 1 cannot read numpy 2 pickles.

## Install

The setup script clones the three upstream projects next to this repo, builds one conda env, and installs everything into it:

```bash
mkdir grace2ace && cd grace2ace                       # the workspace
git clone <this repo> grace2ace
bash grace2ace/scripts/setup_workspace.sh             # env name: grace2ace (or pass another)
```

The workspace then looks like this:

```
grace2ace/                 workspace
  grace2ace/               this package            (editable install)
  assyst/                  eisenforschung/assyst   main            (editable)
  grace-tensorpotential/   ICAMS, tag 0.6.2.1                       (editable)
  python-ace/              ICAMS, feature/grace_fs @ d0f7a0b        (built: C++ extension)
```

- **Pinned versions.** These are tested together. Override them with `ASSYST_REF`, `GRACE_REF` and `PACE_REF`.
- **Compiler note.** pacemaker's bundled yaml-cpp needs `CXXFLAGS="-include cstdint"` with gcc ≥ 13; the script sets it.

## Quick start

```bash
grace2ace init-config local_morse_test -o test.yaml    # no GPU, no GRACE: a Morse potential plays the teacher
# set envs.generate and envs.fit in test.yaml to the env's python, then:
grace2ace run test.yaml --local                        # whole chain in about a minute

grace2ace init-config NiAlW_00 -o NiAlW_00.yaml        # the real thing
# edit system.run_dir, envs.*, slurm.account / partitions, then:
grace2ace run NiAlW_00.yaml                            # submits every stage with dependencies
grace2ace status runs/NiAlW_00                         # progress, fit loss, test errors
grace2ace resume runs/NiAlW_00                         # after a crash: resubmits only what is missing
```

Shipped examples: `NiAlW_00` (the full Ni-Al-W recipe), `NiAlW_smoke` (a tiny check on SLURM with GRACE) and `local_morse_test`.

## The config file

Every key is checked: a misspelled key is an error, not a silent no-op. The sections are:
- **`system`**: name, elements, seed, run directory.
- **`teacher`**: `grace` + model name, or `morse` for tests.
- **`assyst`**: the structure recipe.
  - `range`/`trim` work as in `Formulas.range(...).trim(...)`.
  - Spacegroups and structures per formula.
  - Relaxation steps.
  - Perturbations.
  - Distance radii.
  - `n_shards`.
- **`filters`**: maximum volume per atom, energy window, maximum force.
- **`extra_data`**: other datasets to add (pacemaker pickles or extxyz), e.g. liquid snapshots.
- **`dataset`**: test fraction, and the reference energy (`freeze` or an explicit dict).
  - `reuse_from: <other run_dir>` fits on that run's finished dataset: generate, label and build_df are skipped, and the same training set, test set and reference energy are used. Two fits then differ only in their fit settings. Submission is refused until that run's dataset is done.
- **`fit.input`**: anything here is deep-merged over [`templates/pacemaker_default.yaml`](src/grace2ace/templates/pacemaker_default.yaml); `null` deletes a key.
- **`envs`**: the python of each env.
- **`executor`**: `slurm` (one job per stage, shards in parallel), `slurm_single` (ONE job on one GPU runs every stage in turn, see below) or `local`.
- **`slurm`**: account, setup lines, and partition/time/gpus/cpus per stage (`slurm.single` for `slurm_single`).

**`executor: slurm_single`** suits queues where waiting is the bottleneck: you queue once for one GPU.
- The structure shards run one after another.
- Inside any SLURM job, the fit asks SLURM how much time is left and stops `fit.timeout_margin_minutes` before the end. The job then resubmits itself, and the fit continues from its last interim potential.
- The job does not start a fit with less than `fit.min_start_minutes` left.
- A job never resubmits itself after a real error, and never more than `fit.max_attempts` times.

Perturbations map one-to-one onto ASSYST classes:

```yaml
perturbations:
  - stretch: {hydro: 0.15, shear: 0.02}                                  # Stretch(...)
  - rattle: {sigma: 0.03, create_supercells: true}                        # Rattle(...)
  - [{stretch: {hydro: 0.05, shear: 0.005}}, {rattle: {sigma: 0.05}}]     # Stretch(...) + Rattle(...)
  - random_choice: {a: {rattle: {sigma: 0.25}}, b: {stretch: {hydro: 0.4, shear: 0.05}}, chance: 0.3}
  - {repeat: 4, item: {stretch: {hydro: 0.8, shear: 0.05}}}
```

To change a run after it started, edit `<run_dir>/config.yaml` and call `grace2ace resume <run_dir>`. Finished stages are not redone. Delete a stage's folder to redo it.

## Traps it guards against

Each of these happened in real fits before this tool existed:

- **Silently ignored pacemaker keys.** `fit.loss.L1`/`L2` are dropped without a word (the right names are `L1_coeffs`/`L2_coeffs`), and so is `backend.gpu`. The input is checked against the key lists in pyace 0.2.7's source before the fit starts.
- **Missing `radparameters`** makes pacemaker segfault, even for `SBessel`. It is checked.
- **`nfit` at or below the number of structures under `DElow`** means nothing above `DElow` gets fitted. It is checked with pacemaker's own convex-hull distance.
- **`reference_energy: auto` changes from dataset to dataset**, so energies of two fits are not comparable. grace2ace computes it once, with the same least-squares and shift as pacemaker, and writes it into the input explicitly.
- **A random test split leaks.** Strained and rattled copies of one crystal land in both train and test. grace2ace splits by *family*, meaning all structures grown from the same random crystal.
- **Filters that silently empty a corner of composition space.** Every label shard writes `drop_report_NNN.json`: dropped structures by subsystem and reason.
- **Walltime kills.** pacemaker is stopped shortly before the walltime. The next attempt continues from the latest interim potential with the remaining fit cycles, in its own `fit/attempt_NN` folder.
- **A fit that silently runs on the CPU.** The fit log states which GPUs TensorFlow sees, and warns when an allocated GPU is unusable.
- **Test numbers that are not the final potential's.** `repulsion: auto` adds a ZBL wall *after* the fit, below the shortest distance per bond in the **training** set, and pacemaker's own `test_pred` does not include it. `evaluate` scores the final file that LAMMPS uses. A test cell with atoms closer together than any training cell then shows up, instead of hiding.
- **Unseeded randomness.** Every shard gets its own seeded random stream: the same config gives the same structures.

## What comes out

```
<run_dir>/
  config.yaml            frozen config (edit + `resume` to change)
  generate/  label/      extxyz shards, drop reports, timings in *.done
  build_df/              train.pckl.gzip, test.pckl.gzip, reference_energy.json, summary.json, dataset.png
  fit/                   attempt_NN/ (pacemaker's own files), <name>.yaml, <name>.yace, status.json
  evaluate/              report.json, test_errors.csv.gz, parity.png
  jobs/  logs/           job scripts and SLURM logs
```

## Strategies, and what is next

This version implements the first of these:

1. **Big net (this version).** Many random small crystals, shaken and stretched, all labelled by the teacher.
2. **Pick the useful ones.** Labelling is cheap: GRACE labels 125,000 structures in about 9 minutes. The fit is expensive: about 33 GPU-hours. So label everything, then keep only structures that are different from each other, using `pace_select` (D-optimality on ACE's own descriptors). Same knowledge, smaller training set, faster fit. *Planned as a `select` stage.*
3. **Teacher checks the homework (active learning).**
   - Run ACE MD where the potential will be used: high T, liquid, defects.
   - Collect frames where ACE is unsure (extrapolation grade γ from `pace_activeset`), or where ACE and GRACE disagree. The teacher is cheap, so the *true* error can be measured.
   - Label those frames, add them, and refit starting from the previous potential.

   *Planned as an `explore` stage.*
4. **Ask the teacher during the MD (on-the-fly learning).** Not a good match for pacemaker, whose fits take hours.

## Tests

```bash
python -m pytest            # fast, no GPU; the pyace-dependent check is skipped if pyace is missing
```

## License

BSD-3-Clause, see `LICENSE`.
