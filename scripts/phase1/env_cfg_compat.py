"""Compatibility shims for unitree_sim_isaaclab on Isaac Lab 6.x (legion)."""

from __future__ import annotations

import inspect


def patch_unitree_env_cfgs() -> None:
    """Replace legacy ``sim.physx.*`` usage with Isaac Lab 6 ``sim.physics = PhysxCfg(...)``."""
    try:
        from isaaclab_physx.physics import PhysxCfg
    except ImportError:
        return

    import isaaclab.envs.mdp as base_mdp
    import torch
    from isaaclab.managers import SceneEntityCfg
    from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

    def _make_post_init(decimation: int, episode_length_s: float, reset_y_range):
        def __post_init__(self):
            self.decimation = decimation
            self.episode_length_s = episode_length_s
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            self.sim.physics = PhysxCfg(
                bounce_threshold_velocity=0.01,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
                gpu_total_aggregate_pairs_capacity=16 * 1024,
                friction_correlation_distance=0.00625,
            )
            self.event_manager = SimpleEventManager()
            self.event_manager.register(
                "reset_object_self",
                SimpleEvent(
                    func=lambda env: base_mdp.reset_root_state_uniform(
                        env,
                        torch.arange(env.num_envs, device=env.device),
                        pose_range={"x": [-0.05, 0.05], "y": reset_y_range},
                        velocity_range={},
                        asset_cfg=SceneEntityCfg("object"),
                    )
                ),
            )
            self.event_manager.register(
                "reset_all_self",
                SimpleEvent(
                    func=lambda env: base_mdp.reset_scene_to_default(
                        env, torch.arange(env.num_envs, device=env.device)
                    )
                ),
            )

        return __post_init__

    patched = []
    modules = [
        (
            "tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex3.pickplace_cylinder_g1_29dof_dex3_joint_env_cfg",
            "PickPlaceG129DEX3JointEnvCfg",
            2,
            20.0,
            [-0.05, 0.05],
        ),
    ]
    for mod_name, cls_name, dec, ep_len, y_range in modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            cls.__post_init__ = _make_post_init(dec, ep_len, y_range)
            patched.append(cls_name)
        except Exception as exc:
            print(f"[compat] skip {cls_name}: {exc}")

    if patched:
        print(f"[compat] patched env cfgs: {', '.join(patched)}")
