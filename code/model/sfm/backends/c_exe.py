"""PanicPackage C exe backend (sd_crunch_scenario.exe / sd_crunch_stall.exe)."""
import os
import glob
import shutil
import tempfile
import time
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAR_FILE     = os.path.join(_ROOT, 'PanicPackage', 'sd.par')
# Platform-aware executable name: Windows builds carry the .exe suffix,
# macOS/Linux builds do not. Physics/binary are otherwise identical.
_EXE_EXT     = '.exe' if os.name == 'nt' else ''
EXE_SCENARIO = os.path.join(_ROOT, 'PanicPackage', 'sd_crunch_scenario' + _EXE_EXT)
EXE_STALL    = os.path.join(_ROOT, 'PanicPackage', 'sd_crunch_stall' + _EXE_EXT)
# Absolute dir holding the precomputed bend floor-field CSVs (ff_field_a*.csv).
# Exported to the C exe via SFM_BEND_FIELD_DIR so ff_load_file() can locate them
# regardless of the (temp) working directory the exe is launched in.
BEND_FIELD_DIR = os.path.join(_ROOT, 'results', 'bend_study')


def load_par_template():
    with open(PAR_FILE) as fh:
        return fh.readlines()


def _patch(original_lines, overrides):
    """Patch sd.par lines with key->value overrides. Keys match sd.par tokens."""
    # Map of "starts-with token" -> sd.par var name suffix to detect line
    out = []
    for line in original_lines:
        replaced = False
        for k, v in overrides.items():
            # Build the prefix used in sd.par (e.g. "# i 0 N0", "# f 1 V0")
            for prefix in (f"# i 0 {k}", f"# f 0 {k}", f"# i 1 {k}", f"# f 1 {k}"):
                if line.startswith(prefix):
                    out.append(f"{prefix}\t{v}\n")
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            out.append(line)
    # Additive diagnostic: warn (do not fail) if an override never matched an
    # sd.par token — previously these were dropped silently. Behaviour of the
    # patched file is unchanged; this only prints a warning.
    matched = set()
    for line in out:
        for k in overrides:
            for prefix in (f"# i 0 {k}", f"# f 0 {k}", f"# i 1 {k}", f"# f 1 {k}"):
                if line.startswith(prefix + "\t"):
                    matched.add(k)
    for k in overrides:
        if k not in matched:
            print(f"  [c_exe] WARNING: override '{k}' did not match any sd.par "
                  f"token — it was NOT applied.", flush=True)
    return out


def _run_exe(tmp_par, tmp, exe, timeout_s=120):
    """Run sd_crunch_scenario.exe with hard timeout. Returns dat/dat2/elapsed.
    Kills the C process after timeout_s seconds to prevent runaway sims."""
    t0 = time.time()
    # Tell the C exe where the bend floor-field CSVs live (absolute path),
    # since it is launched with cwd=<tempdir>. No effect on non-bend runs.
    # Additive: honour a caller-set SFM_BEND_FIELD_DIR (e.g. the 5 m field set)
    # and fall back to the default 3 m directory when it is not set.
    _field_dir = os.environ.get('SFM_BEND_FIELD_DIR', BEND_FIELD_DIR)
    child_env = dict(os.environ, SFM_BEND_FIELD_DIR=_field_dir)
    try:
        # NOTE: on Windows, stdout=stderr=DEVNULL with this C exe sometimes
        # hangs the child. capture_output works reliably (output is tiny).
        subprocess.run([exe, '0', tmp_par], cwd=tmp,
                       capture_output=True, timeout=timeout_s, env=child_env)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.time() - t0
    dat   = glob.glob(os.path.join(tmp, 'sd.dat_RndSeed=*_VSmoke=*_N0=*_'))
    dat2  = glob.glob(os.path.join(tmp, 'sd.dat2_RndSeed=*_VSmoke=*_N0=*_'))
    if timed_out:
        print(f'  [c_exe] TIMEOUT after {elapsed:.0f}s (kept partial files)', flush=True)
    return (dat[0] if dat else None,
            dat2[0] if dat2 else None,
            elapsed)


def run_with_trajectory(case, v0, seed, *, traj_dt, max_sim_time=None,
                        overrides=None, exe=None, timeout_s=300, full_traj=False,
                        stats=None):
    """Run ONE simulation and capture per-frame agent positions every
    `traj_dt` sim-seconds via the C exe trajectory writer. Returns
    `(times, agents_final, frames, elapsed)` where `frames` is a list of
    (t, agents) tuples (same agent-tuple layout as read_dat2).

    This replaces the previous "re-run from t=0 for every snapshot" pattern,
    cutting sweep wall time by ~snap_count×."""
    from .. import metrics as _m

    exe = exe or EXE_SCENARIO
    base = dict(
        N0           = case['params']['N0'],
        V0           = v0,
        RndSeed      = seed,
        ScenarioID   = case['scenario_id'] if case['scenario_id'] is not None else 0,
        ColumnSwitch = 0,
        InjurySwitch = 1,
        DoorWidth    = case['params'].get('DoorWidth', 1.0),
        DefaultDeltaT = case['params'].get('DefaultDeltaT', 0.01),
        V_ChangeLimit = case['params']['V_ChangeLimit'],
        C_NS          = 0.95,
        MaxSimTime    = max_sim_time if max_sim_time is not None
                        else case['params']['MaxSimTime'],
        TrajSaveDt    = float(traj_dt),
    )
    for opt in ('RoomXSize', 'RoomYSize', 'Dmean', 'deltaD', 'Tau',
                'A', 'B', 'C_Young', 'Kappa', 'Gamma', 'GaTh',
                'FCrush_over_1m', 'AnisotropyLambda', 'LaneBias',
                'InflowRate', 'ChamferLen', 'StallKickEnable',
                'DiagonalChamfer', 'CurvedChamferR', 'TStemHalfWidth',
                # additive: bend-study + soft-casualty params (forward if present)
                'InjurySwitch', 'UseFloorField', 'BendAngle', 'BendCornerR',
                'BendWidth', 'CasualtyForceScale', 'CasualtyContactScale',
                'CasualtyRadiusScale', 'CasualtyFrictionScale'):
        if opt in case['params']:
            base[opt] = case['params'][opt]
    if overrides:
        base.update(overrides)

    template = load_par_template()
    tmp = tempfile.mkdtemp(suffix=f'_traj_s{seed}_v{v0}')
    try:
        tmp_par = os.path.join(tmp, 'sd.par')
        with open(tmp_par, 'w') as fh:
            fh.writelines(_patch(template, base))
        dat, dat2, elapsed = _run_exe(tmp_par, tmp, exe, timeout_s=timeout_s)
        times  = _m.read_exit_times(dat) if dat else {}
        agents = _m.read_dat2(dat2)      if dat2 else []
        traj_glob = glob.glob(os.path.join(tmp, 'sd.traj_RndSeed=*_VSmoke=*_N0=*_'))
        if traj_glob:
            frames = (_m.read_trajectory_full(traj_glob[0]) if full_traj
                      else _m.read_trajectory(traj_glob[0]))
            if stats is not None:                 # controlled-inflow spawn footer
                stats.update(_m.read_spawn_footer(traj_glob[0]))
        else:
            frames = []
        return times, agents, frames, elapsed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_one(case, v0, seed, *, overrides=None, exe=None, max_sim_time=None,
            timeout_s=120):
    """Run one (case, v0, seed) trial. Returns (exit_times, agents, elapsed).

    timeout_s: kill C process if it exceeds this wall time (default 120 s).
    Set higher for high-density bidir, lower for quick smoke tests."""
    from .. import metrics as _m

    exe = exe or EXE_SCENARIO
    base = dict(
        N0           = case['params']['N0'],
        V0           = v0,
        RndSeed      = seed,
        ScenarioID   = case['scenario_id'] if case['scenario_id'] is not None else 0,
        ColumnSwitch = 0,
        InjurySwitch = 1,
        DoorWidth    = case['params'].get('DoorWidth', 1.0),
        DefaultDeltaT = case['params'].get('DefaultDeltaT', 0.01),
        V_ChangeLimit = case['params']['V_ChangeLimit'],
        C_NS          = 0.95,
        MaxSimTime    = max_sim_time if max_sim_time is not None
                        else case['params']['MaxSimTime'],
    )
    # Forward ALL case-specific sd.par-known params (Moussaid C_Young=40, etc.)
    for opt in ('RoomXSize', 'RoomYSize', 'Dmean', 'deltaD', 'Tau',
                'A', 'B', 'C_Young', 'Kappa', 'Gamma', 'GaTh',
                'FCrush_over_1m', 'AnisotropyLambda', 'LaneBias',
                'InflowRate', 'ChamferLen', 'TrajSaveDt', 'StallKickEnable',
                'DiagonalChamfer', 'CurvedChamferR', 'TStemHalfWidth'):
        if opt in case['params']:
            base[opt] = case['params'][opt]
    if overrides:
        base.update(overrides)

    template = load_par_template()
    tmp = tempfile.mkdtemp(suffix=f'_s{seed}_v{v0}')
    try:
        tmp_par = os.path.join(tmp, 'sd.par')
        with open(tmp_par, 'w') as fh:
            fh.writelines(_patch(template, base))
        dat, dat2, elapsed = _run_exe(tmp_par, tmp, exe, timeout_s=timeout_s)
        times  = _m.read_exit_times(dat) if dat else {}
        agents = _m.read_dat2(dat2)      if dat2 else []
        return times, agents, elapsed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
