"""Isaac Lab 6.x compatible observation helpers for unitree_sim."""

from __future__ import annotations

import torch


def _as_torch_device(dev) -> torch.device:
    if isinstance(dev, torch.device):
        return dev
    return torch.device(str(dev))


def _as_torch_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.float32


def boy_states(env, enable_dds: bool = True):
    import tasks.common_observations.g1_29dof_state as g1s

    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    device = _as_torch_device(joint_pos.device)
    dtype = _as_torch_dtype(joint_pos.dtype)
    batch = joint_pos.shape[0]

    cache = g1s._obs_cache
    if cache["device"] != device or cache["boy_idx_t"] is None:
        boy_joint_indices = [
            0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
            21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
        ]
        cache["boy_idx_t"] = torch.tensor(boy_joint_indices, dtype=torch.long, device=device)
        cache["device"] = device
        cache["batch"] = None

    idx_t = cache["boy_idx_t"]
    n = idx_t.numel()
    if cache["batch"] != batch or cache["boy_idx_batch"] is None:
        cache["boy_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["combined_buf"] = torch.empty(batch, n * 3, device=device, dtype=dtype)
        cache["batch"] = batch

    idx_batch = cache["boy_idx_batch"]
    pos_buf, vel_buf, torque_buf, combined_buf = (
        cache["pos_buf"],
        cache["vel_buf"],
        cache["torque_buf"],
        cache["combined_buf"],
    )
    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    combined_buf[:, 0:n].copy_(pos_buf)
    combined_buf[:, n : 2 * n].copy_(vel_buf)
    combined_buf[:, 2 * n : 3 * n].copy_(torque_buf)
    return combined_buf


def dex3_states(env, enable_dds: bool = True):
    import tasks.common_observations.dex3_state as dex3s

    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    device = _as_torch_device(joint_pos.device)
    dtype = _as_torch_dtype(joint_pos.dtype)
    batch = joint_pos.shape[0]

    cache = dex3s._obs_cache
    if cache["device"] != device or cache["hand_idx_t"] is None:
        gripper_joint_indices = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
        cache["hand_idx_t"] = torch.tensor(gripper_joint_indices, dtype=torch.long, device=device)
        cache["device"] = device
        cache["batch"] = None

    idx_t = cache["hand_idx_t"]
    n = idx_t.numel()
    if cache["batch"] != batch or cache["hand_idx_batch"] is None:
        cache["hand_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=dtype)
        cache["batch"] = batch

    idx_batch = cache["hand_idx_batch"]
    pos_buf, vel_buf, torque_buf = cache["pos_buf"], cache["vel_buf"], cache["torque_buf"]
    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))
    return pos_buf


def patch_observation_device_compat() -> None:
    import tasks.common_observations.g1_29dof_state as g1s
    import tasks.common_observations.dex3_state as dex3s

    g1s.get_robot_boy_joint_states = boy_states
    dex3s.get_robot_dex3_joint_states = dex3_states
    print("[compat] patched observation module helpers")
