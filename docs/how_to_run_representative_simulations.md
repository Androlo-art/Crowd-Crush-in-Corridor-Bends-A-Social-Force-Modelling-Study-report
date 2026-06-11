# How to run a representative simulation

**Re-running simulations is not required to reproduce the figures** (regenerate those
from the included `results/` — see `how_to_regenerate_figures.md`). Running simulations
is only needed if you want to reproduce the raw data itself.

⚠️ **This is not quick.** The thesis sweeps are large (e.g. the coupled study is
6 angles × 8 exit widths × 15 seeds = 720 runs; the obstacle study is 8 layouts × 6
angles). The full sweeps are hundreds of CPU-hours. The steps below describe a single
**representative** run, not the full reproduction.

## 1. Obtain and build the simulation core

The C escape-panic core is **third-party and is not included in this repository** (its
licence terms are under review). Obtain the base package from the upstream source and
apply the thesis modifications as described in `code/model/c_core/README.md`, then build
it with a C compiler. The Python driver (`code/model/sfm/`) calls the resulting binary.

## 2. Generate the floor field for the geometry you want

The simulation reads a precomputed fast-marching navigation field per geometry. Use the
builders in `code/model/navigation/` (e.g. `make_bend_field.py`) to produce the field
folder the runner expects (one folder per exit width / radius). A precomputed field set
for the d=1.2 m chamfered geometry is included at
`results/00_model/navigation_fields_d12/` for the navigation figure.

## 3. Run one representative cell

Example (one coupled turn/exit cell, a single seed). The runners in
`code/experiments/<section>/` drive the C binary via the `sfm.backends.c_exe` wrapper.
Open the script header for the exact flags; each exposes `--seeds`, `--angles`,
`--reduced` (smoke) and `--exe`. For instance, a reduced smoke run of the
open/restricted bend study:

```bash
cd code/experiments/02_open_and_restricted_bend
python run_open_and_restricted_bend.py --reduced --jobs 4          # restricted (bottleneck) binary
python run_open_and_restricted_bend.py --reduced --jobs 4 \
       --exe ../../model/c_core/bend/sd_crunch_openbend             # open-bend comparison
```

Outputs are written under `results/<...>/` with a fresh timestamped folder plus a
`run_info.txt` recording every parameter.

## Faster-is-slower

Representative faster-is-slower simulations may be run from the model code after building
the room binary from the upstream escape-panic package (see `code/model/c_core/README.md`) and driving it from the
`sfm` driver — a single-room evacuation over the desired-speed grid reproduces the
clogging behaviour. The exact published FIS sweeps are not reproduced from this archive;
the final raw result tables behind the FIS figures are included in
`results/01_faster_is_slower/`.

## Caveats
- The run scripts' field/output directory paths point into the release tree but assume
  you have built the binary (step 1) and generated the matching floor field (step 2);
  adjust the `EXE` / `FIELD_DIR` constants at the top of each runner if your layout differs.
- Results are deterministic in `(parameters, seed)`; parallelism only changes wall-clock,
  not the numbers.
