"""Load teleop tuning from config/teleop.toml.

CLI flags override the file; the file overrides the code defaults.  Uses the
stdlib tomllib (Python 3.11+), so there is no extra dependency.
"""

import os

try:
    import tomllib
except ModuleNotFoundError:                      # pragma: no cover
    tomllib = None

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config", "teleop.toml")

# (toml section, toml key) -> argparse dest
MAPPING = {
    ("motion", "gain"): "gain",
    ("motion", "max_step"): "max_step",
    ("motion", "speed"): "speed",
    ("workspace", "max_reach"): "max_reach",
    ("workspace", "min_z"): "min_z",
    ("rotation", "unlock"): "unlock_rotation",
    ("rotation", "gain"): "rot_gain",
    ("rotation", "deadband"): "rot_deadband",
    ("rotation", "max_step"): "max_rot_step",
    ("safety", "max_joint_step"): "max_joint_step",
    ("safety", "lead_limit"): "lead_limit",
    ("keyboard", "step"): "step",
    ("keyboard", "rot_step"): "rot_step",
    ("quest", "port"): "port",
    ("quest", "frame_mode"): "frame_mode",
}


def load(path=CONFIG_PATH, keys=None):
    """Return {argparse_dest: value} for the keys this script understands."""
    if tomllib is None or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    out = {}
    for (section, key), dest in MAPPING.items():
        if section in raw and key in raw[section]:
            if keys is None or dest in keys:
                out[dest] = raw[section][key]
    return out


def describe(path=CONFIG_PATH):
    return path if os.path.exists(path) else "(no config file; using code defaults)"
