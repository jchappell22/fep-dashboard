# fep-dashboard

A web UI over the shared OpenFE and TMD installs on Conifer, so running a
free-energy campaign is a form and a progress page rather than a tmux
session and a folder of TSVs.

It does four things:

- **Prepare inputs** — what OpenFE and TMD actually require from a protein
  and a ligand series, plus a checker that reads your files and reports what
  would fail.
- **Launch** — pick engine, method, protein, ligands, GPUs; see the exact
  command that will run; start it.
- **Runs** — live per-leg progress, ETA, logs, kill, resume.
- **Results** — ranked ligands, per-edge ΔΔG, and a network-consistency
  check that tells you which edges not to trust.

It wraps the shared `/opt` installs documented in the OpenFE/TMD install
README — `openfe` and `tmd-submit` in `/usr/local/bin`. It does not vendor,
patch, or reimplement either engine.

---

## The one design decision worth knowing

**Every command this dashboard runs lives in a TOML file, not in Python.**

`engines/openfe.toml` and `engines/tmd.toml` hold the literal command
templates for each engine's plan / run / gather stages. The dashboard
renders them into a `driver.sh` per campaign and spawns that.

This matters because engine CLIs drift. When a flag changes, the fix is one
line in a TOML on the box — not a code change, not a redeploy. It also means
the Launch page can show you the real command *before* you commit a GPU for
twelve hours, and a failed campaign leaves behind a script you can read,
edit, and re-run by hand with no dashboard involved.

---

## Install

```bash
git clone <this repo> /opt/fep-dashboard      # or anywhere on the box
cd /opt/fep-dashboard

mamba env create -f environment.yml            # creates `fep-dash`
mamba activate fep-dash

cp config.example.toml config.toml
$EDITOR config.toml                            # set runs_root and allowed GPUs
```

Prefer a venv? `python -m venv .venv && . .venv/bin/activate && pip install
-r requirements.txt` works too — `scripts/run_ui.sh` finds either. If you
name the conda env something other than `fep-dash`, set `FEPDASH_ENV`.

The `fep-dash` env is deliberately tiny — streamlit, pandas, numpy, and
nothing else. Even the openfe settings YAML is emitted without PyYAML, so a
missing package can never take the dashboard down on a box where installing
one is a chore. It
**never imports openfe or tmd**; it shells out to the `/usr/local/bin`
launchers, which activate their own `/opt` conda envs internally. That is
what lets one small UI env drive two mutually incompatible engine stacks.

## Run

```bash
bash scripts/run_ui.sh          # http://localhost:8578
```

Bound to loopback: the box is shared, there is no auth, and unlike a
read-only status page this one can start GPU jobs. Reach it from the box's
desktop, or forward the port:

```bash
ssh -L 8578:localhost:8578 <user>@conifer
```

## Try it without burning GPU time

```bash
python -m fepdash.core.fixtures --runs-root ~/fep-runs
```

Fabricates campaigns covering the states that actually break things — legs
done, running, pending, and failed at once; a failed leg with a real CUDA
traceback; result tables in both engines' dialects; an edge network with a
deliberate cycle-closure outlier. Nothing touches a GPU. Do this first.

---

## Layout

```
engines/            THE COMMAND TEMPLATES -- edit these to fix a flag
  openfe.toml
  tmd.toml
src/fepdash/
  app.py            home / status board
  pages/            Prepare inputs, Launch, Runs, Results
  core/
    prep_check.py   dependency-free PDB/SDF input checks
    plan_settings.py builds openfe's planning YAML from the Launch form
    driver.py       renders the campaign's driver.sh
    engines/base.py loads + renders the engine TOMLs
    launcher.py     detached spawn, kill, resume
    polling.py      notices a driver exited; writes the transition
    state.py        derives leg progress from the filesystem
    results.py      normalises both engines' tables into one shape
    gpu.py          GPU inventory + cross-engine claims
    fixtures.py     synthetic campaigns
scripts/
  run_ui.sh
  openfe_setup_abfe.py   ABFE setup shim (OpenFE has no plan-abfe CLI)
```

### A campaign directory

```
runs/<campaign_id>/
  campaign.json     the manifest -- engine, method, inputs, params, GPUs
  driver.sh         the exact script that was spawned
  stdout.log        driver output; the Runs page tails this
  plan/             transformations/*.json (OpenFE) or map.json
  legs/<leg>/       status.json + leg.log, written by the driver
  work/<leg>/       engine scratch
  results/          engine result artifacts
  gathered/         ddg.tsv / dg.tsv
  .stage            current stage, one line
```

State lives in two places on purpose. The sqlite DB holds **only** what the
filesystem cannot tell us — the pid, the operator's intent, the exit code.
Everything else is derived by scanning the campaign directory on every read.
So a leg that finished while the dashboard was down still shows as done, two
browser tabs never disagree, and deleting a campaign directory cannot leave
the DB describing runs that no longer exist.

---

## Engines

| | OpenFE | TMD |
|---|---|---|
| reached via | `openfe` (`/opt/openfe/current`) | `tmd-submit` (`/opt/tmd/current`) |
| methods | RBFE, ABFE | RBFE (network), single edge |
| version pin | `OPENFE_VERSION` dropdown on the form | flip `/opt/tmd/current` |
| GPU choice | **dashboard picks**, one leg per card | **launcher picks** the most-free card |
| parallelism | driver fans legs out over `CUDA_VISIBLE_DEVICES` | one process, schedules itself |
| result tables | a `gather` step writes them | written during the run |

### Why there is no GPU picker for TMD

`tmd-submit` auto-selects the most-free GPU. Offering a picker that the
engine ignores would be worse than offering none, and exporting
`CUDA_VISIBLE_DEVICES` to override it would defeat the sharing behaviour
everyone else on the box relies on. So for TMD the dashboard shows GPU state
for awareness and stays out of the way.

### Why the driver never passes `--bg`

The install README's example uses `tmd-submit network ... --bg`, which is
right at an interactive prompt. Here it would be a bug: the dashboard has
already detached the driver and tracks the campaign by that pid. If
`tmd-submit` forked into the background, the tracked pid would exit within
seconds and every campaign would report "finished" while the real work ran
on invisibly.

### Network planning is configured in the UI, not a YAML

`openfe plan-rbfe-network -s` takes a YAML choosing the atom mapper, the
network planner, and the partial-charge method. Nobody hand-writes it: the
Launch page renders widgets, and the dashboard generates the file into the
campaign as `plan_settings.yaml` — so it doubles as the permanent record of
how that network was planned.

The options live in `engines/openfe.toml` under `[rbfe.settings]`, same as
every other engine detail. Adding a planner openfe gained last week is a
`[[rbfe.settings.network.choices]]` entry, no code change.

Three things this handles that a hand-written YAML does not:

- **Kartograf's hydrogen flag.** openfe applies a mapper section as
  `cls(**settings)`, so any key you omit falls back to the *class* default.
  But openfe's own no-YAML path sets `map_hydrogens_on_hydrogens_only=True`,
  which its source calls a "non-default setting". Supplying a mapper section
  and omitting that key therefore silently changes mapping behaviour versus
  plain `openfe`. Every non-optional key is emitted explicitly to avoid it.
- **Optional keys are omitted, never nulled.** openfe `setattr`s whatever it
  reads, so a literal `None` sets the *string* `"None"` — which is what
  openfe's own `DEFAULT_YAML` docstring mistakenly shows. Blank optional
  fields drop the key instead.
- **`generate_radial_network` blocks without a hub.** `central_ligand` has
  no default; leaving it blank kills planning *after* charge generation, the
  slow part. Pre-flight refuses the launch.

Charge methods needing OpenEye, `openff-nagl`, or `espaloma_charge` are
shown but marked unusable — the `/opt` install README states OpenEye is not
licensed here. `am1bcc` + `ambertools` is the verified combination, and it
is also what matches TMD's `smirnoff_2_2_1_amber_am1bcc`, which is what
makes the two engines comparable on the same series.

### Input checks are structural, not chemical

The Prepare page reads PDB and SDF files as **text** — the dashboard env has
no RDKit, and adding one would defeat the point of keeping it installable
anywhere. So it catches the mistakes that kill a campaign in its first
minutes (a ligand left in the protein PDB, altLocs, 2D coordinates, missing
hydrogens, duplicate molecule names, mixed net charge) and explicitly does
not judge tautomers, protonation states, or perceived bond orders.

Severity is calibrated so the checker stays worth listening to: ERROR means
it will fail or give a meaningless answer, WARN means look before spending
cards. An unrecognised residue in ATOM records is only a WARN, because the
built-in residue list cannot be exhaustive — a correctly capped C-terminus
is `NME` to Amber and `NMA` to other tools, and hard-failing on that would
flag properly prepared proteins.

Validated against real inputs: it passes the prepped protein from the ST4
campaign and flags the un-prepped version of the same structure for
alternate conformations.

### GPU claims

Launching an OpenFE campaign claims its cards; the claim is released when
the owning process dies, swept on read, so a killed dashboard or a rebooted
box self-heals. Claims are **advisory** — they cannot see jobs started
outside the dashboard, which is why the Launch page also shows live
`nvidia-smi` state. Sanity-check it before a long run, and never kill
foreign jobs.

---

## Verified vs. unverified

Built on a laptop with no access to Conifer, so be precise about this.

**Verified** — exercised and passing:

- Both engine TOMLs load; all four driver variants render and pass `bash -n`.
- A generated OpenFE driver **executed end to end** against a stub `openfe`:
  planning, two-GPU fan-out with atomic claims, a deliberately failing leg,
  gather. Then re-run, and it correctly skipped completed legs, released the
  stale claim, and retried only the failure.
- OpenFE CLI flags (`plan-rbfe-network`, `quickrun`, `gather`, `gather-abfe`)
  checked against a real openfe **1.10.0**.
- Every settings attribute `openfe_setup_abfe.py` touches, checked against
  that same install.
- Results parsing, leg scanning, ETA, poller, and cycle-closure exercised
  against generated fixtures in both TSV and CSV dialects.
- **The generated planning YAML, against openfe itself.** Every mapper ×
  network combination the TOML declares is fed to openfe's own
  `load_yaml_planner_options`, which does the registry lookups, constructs
  the mapper as `cls(**settings)`, and builds `partial(func, **settings)`.
  A misspelled method key or a kwarg that no longer exists fails there. Run
  it in an OpenFE env: `/opt/openfe/current/bin/python -m pytest
  tests/test_plan_settings.py -q` (it skips where openfe is absent).

**Unverified** — check these before the first real campaign:

- **`tmd-submit`'s actual flags.** Transcribed from the install README.
  `network` and `edge` and their `--sdf/--pdb/--a/--b/--out` come from its
  examples; whether `edge` accepts `--out`, and the exact filenames written
  under `--out`, are assumptions. Run `tmd-submit --help` and reconcile
  `engines/tmd.toml`.
- **Where TMD writes its result tables.** This is the most likely remaining
  *silent* failure. `engines/tmd.toml` sets `tables_in = "results_dir"` and
  expects `ddg_results.csv` / `dg_results.csv` directly under `--out`. But
  TMD's tutorial mentions "per-ligand directories containing simulation
  plots and results" — if `--out` gets subdirectories instead, the Results
  page will say "no result tables yet" forever on a campaign that actually
  succeeded. Fix by pointing `ddg_table` / `dg_table` at the real relative
  path once you have seen one.
- **TMD's result table names** themselves come from the TMD project's
  tutorials, not from this box.
- **OpenFE 1.11.1/1.12.0 flags**, if they moved since 1.10.0.
- **TMD ABFE** is deliberately not offered. The repo ships an ABFE example,
  but the launcher documents only `edge` and `network`, so there is no
  verified way to drive it. Add an `[abfe.*]` section to `engines/tmd.toml`
  once you know the invocation.

The Launch page renders the full command and `driver.sh` before anything is
spawned, so a mismatch shows up in seconds rather than after a long run.

---

## Fixing a wrong command

1. Open `engines/<engine>.toml`.
2. Fix the `cmd` template.
3. Reload the Launch page — no restart needed.

Placeholders are `{name}` substitution only. An unknown placeholder is a
hard error at render time, never a silently empty string, because a
silently-dropped `--pdb` is a twelve-hour run that produces garbage.
