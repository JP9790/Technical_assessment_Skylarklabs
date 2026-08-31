"""Bootstrap Isaac Lab imports before loading unitree_sim_isaaclab tasks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
UNITREE_SIM = JP_TEST_ROOT / "external" / "unitree_sim_isaaclab"
ISAACLAB = Path("/home/autonomique/AVSR/IsaacLab")
STUBS = Path(__file__).resolve().parent / "stubs"

os.environ.setdefault("PROJECT_ROOT", str(UNITREE_SIM))

_PATHS = [
    str(STUBS),
    str(ISAACLAB / "source" / "isaaclab"),
    str(ISAACLAB / "source" / "isaaclab_tasks"),
    str(ISAACLAB / "source" / "isaaclab_rl"),
    str(ISAACLAB / "source" / "isaaclab_assets"),
    str(UNITREE_SIM),
]
for p in reversed(_PATHS):
    if p not in sys.path:
        sys.path.insert(0, p)


def patch_configclass() -> None:
    from isaaclab.utils.configclass import configclass as _configclass_fn
    import isaaclab.utils as utils_mod

    utils_mod.configclass = _configclass_fn  # type: ignore[attr-defined]


def import_unitree_tasks():
    patch_configclass()
    import tasks  # noqa: F401
    from env_cfg_compat import patch_unitree_env_cfgs
    from observation_compat import patch_observation_device_compat

    patch_unitree_env_cfgs()
    patch_observation_device_compat()
    return tasks
