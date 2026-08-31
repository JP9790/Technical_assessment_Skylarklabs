"""Action helpers for Phase 1 scripted control."""

from __future__ import annotations

import torch


def action_from_target(env, target_full: torch.Tensor) -> torch.Tensor:
    """Map absolute joint targets to JointPositionAction offsets (action-manager order)."""
    default = env.scene["robot"].data.default_joint_pos[0]
    term = env.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids
    if joint_ids == slice(None):
        offset = target_full - default
    else:
        offset = target_full[joint_ids] - default[joint_ids]
    return offset.unsqueeze(0)
