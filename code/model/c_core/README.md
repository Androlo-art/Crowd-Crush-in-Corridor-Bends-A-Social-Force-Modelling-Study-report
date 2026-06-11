# Simulation core (third-party)

The simulation core is based on the pedestrian escape-panic package of Farkas, Helbing and
Vicsek (D. Helbing, I. Farkas & T. Vicsek, *Simulating dynamical features of escape panic*,
Nature **407**, 487–490, 2000).

**Upstream source:** https://github.com/fij/panic

The base C source is obtained from the upstream package above. It is third-party and is not
relicensed here; the original non-profit and commercial licence texts are in
`../../../licenses/`.

## Thesis modifications to the base model

This work modifies the base escape-panic model to add:

- a parameterised bend geometry (corridor turning through 0–135°);
- a variable exit width (a parameterised door at the bend outlet);
- a rounded inner corner (a chamfer of selectable radius);
- an optional column / obstacle contact force (a placed pillar or gate, using the same
  contact law as the walls);
- thesis-specific scenario configurations (the room-evacuation benchmark and the
  bend / coupling / obstacle studies, with their parameter sweeps).

All other model dynamics (the social-force law, time integration and crush rule) are
unchanged from the base package.

## What this repository includes

The thesis-specific Python driver (`../sfm/`), the navigation floor-field builders
(`../navigation/`), the analysis and plotting code (`../../plotting/`), the processed
result tables, the report figures and representative videos are included — enough to
inspect and regenerate the reported figures from the processed results.
