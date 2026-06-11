# Model

The simulation model used throughout the thesis: a Helbing–Farkas–Vicsek social-force
model with body contact, wall/column contact, fast-marching navigation, controlled
inflow, and a contact-load crush rule.

```
model/
├── c_core/        README only — the C escape-panic core is third-party and is NOT
│                  redistributed here; see c_core/README.md for the upstream source
│                  and the thesis modifications
├── navigation/    fast-marching (Eikonal) floor-field + corridor geometry builders
│   ├── floorfield_fmm.py       distance field via skfmm, desired direction = -∇D/|∇D|
│   ├── build_bend_geometry.py  corridor polygon / wall geometry
│   └── make_bend_field.py      writes per-geometry navigation fields the C core reads
└── sfm/           thin Python driver that launches the C binary and collects output
    ├── geometry.py, cases.py   case/geometry definitions used by the runners
    └── backends/c_exe.py       runs the compiled C executable (the only backend shipped)
```

Notes
- The corridor studies call the C binary directly through `sfm.backends.c_exe`.
- The model law, integration and crush rule are identical across all studies; only the
  corridor geometry and demand differ. Full parameter values are in the thesis appendix.
- The C escape-panic core is third-party and is **not included** here; see
  `code/model/c_core/README.md` for the upstream source and the thesis modifications.
  To generate navigation fields: the builders in `navigation/`.
