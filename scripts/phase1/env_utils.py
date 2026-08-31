"""Shared env cfg preparation for Phase 1 scripts."""

from __future__ import annotations

from observation_compat import boy_states, dex3_states


def _noop_camera_image(env, **kwargs) -> dict:
    return {}


def prepare_env_cfg(env_cfg, *, use_cameras: bool = True):
    """Apply Isaac Lab 6.x compatibility patches to a parsed env cfg."""
    obs = env_cfg.observations.policy
    obs.robot_joint_state.func = boy_states
    obs.robot_gipper_state.func = dex3_states
    if not use_cameras:
        import tasks.common_observations.camera_state as camera_state

        camera_state.get_camera_image = _noop_camera_image
    return env_cfg
