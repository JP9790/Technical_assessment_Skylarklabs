"""Standalone BrainCo hand FK via Isaac Sim (one session, many queries)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]

TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class BraincoHandFK:
    """Kinematic queries for a single BrainCo hand articulation."""

    def __init__(self, side: str, device: str) -> None:
        import sys

        sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
        sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase3"))
        from brainco_hand_scene import make_hand_scene_cfg
        from isaac_bootstrap import patch_configclass

        patch_configclass()
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationCfg, SimulationContext

        self.side = side
        self.device = device
        cfg = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
        self.actuated: list[str] = list(cfg[f"{side}_hand"]["actuated_joint_names"])
        self.base_link: str = str(cfg[f"{side}_hand"]["base_link"])
        self.tip_links = cfg[f"{side}_hand"]["fingertip_links"]
        self.limits = [
            (
                float(cfg[f"{side}_hand"]["joint_limits"][j]["lower"]),
                float(cfg[f"{side}_hand"]["joint_limits"][j]["upper"]),
            )
            for j in self.actuated
        ]

        HandCfg = make_hand_scene_cfg(side, with_ground=False)
        self.sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
        self.scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()
        self.scene.reset()
        self.robot = self.scene["robot"]
        self._idx = {n: i for i, n in enumerate(self.robot.joint_names)}
        self._body_idx = {n: i for i, n in enumerate(self.robot.data.body_names)}

    def tips_in_base(self, q: np.ndarray) -> dict[str, np.ndarray]:
        target = self.robot.data.default_joint_pos.clone()
        for j, name in enumerate(self.actuated):
            if name in self._idx:
                target[0, self._idx[name]] = float(q[j])
        self.robot.write_joint_position_to_sim(target)
        self.robot.set_joint_position_target(target)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt=0.005)
        base_idx = self._body_idx.get(self.base_link)
        if base_idx is None:
            for name, idx in self._body_idx.items():
                if "base" in name.lower():
                    base_idx = idx
                    break
            if base_idx is None:
                base_idx = 0
        base = self.robot.data.body_pos_w[0, base_idx].cpu().numpy()
        out: dict[str, np.ndarray] = {}
        for finger, link in self.tip_links.items():
            if link not in self._body_idx:
                continue
            pos = self.robot.data.body_pos_w[0, self._body_idx[link]].cpu().numpy()
            out[finger] = pos - base
        return out

    def bounds(self) -> list[tuple[float, float]]:
        return list(self.limits)
