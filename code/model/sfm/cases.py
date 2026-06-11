"""Case registry — single source of truth for all simulation cases.

A case = geometry + per-scenario sim params + sweep kind + allowed backends.
The runner dispatches via this registry. Adding a new case = edit this file only.
"""
from .geometry import BUILDERS

# Sweep kinds (drives runner branching + plotting)
SK_SINGLE      = 'single'         # one (v0, seed) — heatmaps + animation
SK_FIS         = 'fis'            # v0 × seed sweep — Helbing Fig 1c/d
SK_DOORWIDTH   = 'doorwidth'      # door-width sweep — Moussaid Fig.S5
SK_PARAM_GRID  = 'param_grid'     # dT × VCL × v0 × seed sensitivity grid
SK_CORRIDOR    = 'corridor'       # single v0, multi seeds — density/pressure
SK_DENSITY     = 'density'        # N × seed sweep at fixed v0 — corridor density
SK_BIDIR       = 'bidirectional'  # closed-loop respawn corridor, lane formation

# Backends
B_C    = 'c'        # PanicPackage sd_crunch_*.exe
B_MEX  = 'mex'      # sfm_mex (MATLAB MEX)
B_MAT  = 'matlab'   # pure-MATLAB run_sim (corridors)

# scenario_id used by C exe when applicable
CASES = {
    'fis_room': dict(
        label='FIS Room — Helbing 2000 Fig.1c/1d (15×15 m, east door)',
        geometry='fis_room',
        scenario_id=0,
        sweep_kind=SK_FIS,
        params=dict(N0=200, MaxSimTime=400.0, V_ChangeLimit=0.01, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_v0=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        backends=[B_C, B_MEX],
    ),
    'single_room': dict(
        label='Single Room Run — 15×15 m, one (v0, seed)',
        geometry='fis_room',
        scenario_id=0,
        sweep_kind=SK_SINGLE,
        params=dict(N0=200, MaxSimTime=400.0, V_ChangeLimit=0.01, DoorWidth=1.05),
        default_seeds=[0],
        default_v0=1.5,
        backends=[B_C, B_MEX],
    ),
    'moussaid_room': dict(
        # Moussaid 2011 Fig.S5 — paper params: τ=0.5, φ=90°, dmax=2, mi=60 kg,
        # k=5×10³ N/m, v0=1.4 m/s. k/m ≈ 80, and k_eff = 2·C_Young → C_Young=40.
        # Paper does NOT specify A, Kappa, GaTh — they remain at our Helbing-2000
        # defaults (A=25, Kappa=3000, GaTh=1) unless --minimal_forces is passed.
        label='Moussaid Fig.S5 — 4×10 m room, door-width sweep',
        geometry='moussaid_room',
        scenario_id=0,
        sweep_kind=SK_DOORWIDTH,
        params=dict(N0=80, V0=1.4, MaxSimTime=250.0, V_ChangeLimit=0.05,
                    RoomXSize=10.0, RoomYSize=4.0, Dmean=0.5, deltaD=0.05,
                    Tau=0.5,
                    # Paper-specific body stiffness (overrides Helbing default)
                    C_Young=40.0),
        default_seeds=[0],
        default_doors=[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        backends=[B_C, B_MEX],
    ),
    'l_narrow': dict(
        label='L/T-merge Narrow (20×10 m, 2-m arms)',
        geometry='l_narrow',
        scenario_id=1,
        sweep_kind=SK_CORRIDOR,
        params=dict(N0=50, MaxSimTime=150.0, V_ChangeLimit=0.05, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_v0=1.5,
        backends=[B_C, B_MEX],
    ),
    'l_wide': dict(
        label='L/T-merge Wide (20×10 m, 4-m arms)',
        geometry='l_wide',
        scenario_id=2,
        sweep_kind=SK_CORRIDOR,
        params=dict(N0=80, MaxSimTime=200.0, V_ChangeLimit=0.05, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_v0=1.5,
        backends=[B_C, B_MEX],
    ),
    't_corridor': dict(
        label='T-Corridor (40×12 m, 4-m stem, batch + optional inflow)',
        geometry='t_corridor',
        scenario_id=3,
        sweep_kind=SK_DENSITY,
        params=dict(N0=100, MaxSimTime=80.0, V_ChangeLimit=0.05, DoorWidth=4.0),
        default_seeds=[0, 1, 2],
        default_v0=1.3,
        backends=[B_C, B_MEX],
    ),
    'moussaid_corner': dict(
        label='Moussaid L-turn (16×12 m, MATLAB only)',
        geometry='moussaid_corner',
        scenario_id=None,
        sweep_kind=SK_CORRIDOR,
        params=dict(N0=65, MaxSimTime=120.0, V_ChangeLimit=0.05, DoorWidth=1.5),
        default_seeds=[0, 1, 2],
        default_v0=1.5,
        backends=[B_MAT],
    ),
    'corner_90': dict(
        # ScenarioID=5: 90° corner (Moussaid 2011 Fig.S4 paper geometry).
        # 10×10 m, walkable L = [0,5]×[0,10] + [0,10]×[5,10]. Cut-out at [5,10]×[0,5].
        # Batch start: N0 agents pre-placed across L-shape, all surge toward
        # elbow (5,5), exit through FULL east wall of upper arm (DW=5 m).
        # SK_DENSITY: vary N to map pressure at inner corner vs occupancy.
        label='Corner 90° (paper Fig.S4, 10×10 m L-shape)',
        geometry='moussaid_corner',
        scenario_id=5,
        sweep_kind=SK_DENSITY,
        params=dict(MaxSimTime=80.0, V_ChangeLimit=0.05, DoorWidth=5.0),
        default_seeds=[0, 1, 2],
        default_N=[30, 60, 90],
        default_v0=1.3,
        backends=[B_C],
    ),
    'bidirectional': dict(
        # Moussaid 2011 Fig.S2: l=16 m, w=4 m, N=60 (30 blue + 30 red),
        # τ=0.5, k=5e3 (k_eff/m=83 → C_Young=42), v0=1.3, periodic BC, T=30 s.
        # Paper uses cognitive φ=90°, dmax=10 m (not implemented; basic SFM).
        label='Bidirectional corridor (Moussaid Fig.S2) — 16×4 m, periodic BC',
        geometry='bidir_corridor',
        scenario_id=4,
        sweep_kind=SK_BIDIR,
        params=dict(N0=60, V0=1.3, MaxSimTime=30.0, V_ChangeLimit=0.05,
                    DoorWidth=1.0, C_Young=42.0),
        default_seeds=[0, 1, 2],
        default_v0=1.3,
        backends=[B_C],
    ),

    'density_l_narrow': dict(
        label='L-narrow density sweep — vary N at fixed v0',
        geometry='l_narrow',
        scenario_id=1,
        sweep_kind=SK_DENSITY,
        params=dict(MaxSimTime=200.0, V_ChangeLimit=0.05, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_N=[20, 40, 60, 80, 100],
        default_v0=1.5,
        backends=[B_C],
    ),
    'density_l_wide': dict(
        label='L-wide density sweep — vary N at fixed v0',
        geometry='l_wide',
        scenario_id=2,
        sweep_kind=SK_DENSITY,
        params=dict(MaxSimTime=250.0, V_ChangeLimit=0.05, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_N=[40, 80, 120, 160, 200],
        default_v0=1.5,
        backends=[B_C],
    ),
    'density_t_corridor': dict(
        label='T-corridor density sweep — vary N at fixed v0',
        geometry='t_corridor',
        scenario_id=3,
        sweep_kind=SK_DENSITY,
        params=dict(MaxSimTime=300.0, V_ChangeLimit=0.05, DoorWidth=1.05),
        default_seeds=[0, 1, 2],
        default_N=[50, 100, 150, 200, 250],
        default_v0=1.5,
        backends=[B_C],
    ),

    'param_grid': dict(
        label='Parameter sensitivity grid — dT × VCL × v0 × seed (MEX)',
        geometry='fis_room',
        scenario_id=0,
        sweep_kind=SK_PARAM_GRID,
        params=dict(N0=200, MaxSimTime=400.0, DoorWidth=1.05),
        default_seeds=[0, 1, 2, 3, 4, 5],
        default_v0=[0.5, 1.0, 1.5, 1.75, 2.0, 3.0, 4.0, 5.0, 6.0],
        default_dT=[0.01],
        default_VCL=[0.01],
        backends=[B_MEX],
    ),
}


def get(name):
    if name not in CASES:
        raise KeyError(f"Unknown case '{name}'. Use --list to see registry.")
    return CASES[name]


def listing():
    """One-line summary of every registered case."""
    lines = [f"{'name':<18} {'kind':<11} {'backends':<14} label"]
    lines.append('-' * 80)
    for name, c in CASES.items():
        backends = ','.join(c['backends'])
        lines.append(f"{name:<18} {c['sweep_kind']:<11} {backends:<14} {c['label']}")
    return '\n'.join(lines)
