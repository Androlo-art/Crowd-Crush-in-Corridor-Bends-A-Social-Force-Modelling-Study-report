# Notice — licensing and attribution

This repository supports the submitted thesis. It includes the thesis-specific code,
processed results, report figures and representative videos.

## Simulation core (third-party)

The C simulation core is based on / derived from the Farkas–Helbing–Vicsek (Helbing)
pedestrian escape-panic package (D. Helbing, I. Farkas & T. Vicsek, *Simulating dynamical
features of escape panic*, Nature 407, 487–490, 2000; upstream: https://github.com/fij/panic).

- This panic-derived code is **not relicensed** by the author.
- The original **non-profit** and **commercial** licence texts are included in `licenses/`
  (`panic_LICENSE_nonprofit.txt` and `panic_LICENSE_commercial.txt`). Please consult those
  files for the terms that apply to the panic-derived components.
- The base C source is obtained from the upstream package; `code/model/c_core/README.md`
  documents the source and the thesis modifications.

## Thesis-specific work

The Python plotting, analysis and figure-generation scripts, the documentation, the
processed result tables and the report figures are the author's own work unless otherwise
stated. No MIT, Apache, GPL or other blanket licence is applied to the repository as a whole.

## Reference data

The digitised Helbing et al. (2000) reference curves in
`results/01_faster_is_slower/helbing_reference/` were digitised from the published figures
and are included only as a comparison baseline for the validation benchmark.
