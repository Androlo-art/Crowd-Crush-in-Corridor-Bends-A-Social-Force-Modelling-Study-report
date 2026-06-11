"""Backend dispatch: the studies use the C-executable backend (``c_exe``)."""
from . import c_exe

ALL = {'c': c_exe}


def get(name):
    if name not in ALL:
        raise KeyError(f"Unknown backend '{name}'. Use one of: {list(ALL)}")
    return ALL[name]
