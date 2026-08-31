#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from robot_control import RIGHT_HAND_OPEN, build_target, palm_object_distance, step_hold_kinematic
from isaaclab.app import AppLauncher
parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
import_unitree_tasks()
import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
env.reset()
candidates = [
    ("pos_a", dict(right_shoulder_pitch_joint=1.2, right_shoulder_roll_joint=-0.9, right_shoulder_yaw_joint=0.3, right_elbow_joint=0.75, right_wrist_pitch_joint=-0.4)),
    ("pos_b", dict(right_shoulder_pitch_joint=0.9, right_shoulder_roll_joint=-1.1, right_shoulder_yaw_joint=0.0, right_elbow_joint=1.0, right_wrist_pitch_joint=-0.3)),
    ("neg_a", dict(right_shoulder_pitch_joint=-1.0, right_shoulder_roll_joint=-0.85, right_shoulder_yaw_joint=-0.35, right_elbow_joint=1.25, right_wrist_pitch_joint=-0.55)),
]
out = []
for label, joints in candidates:
    env.reset()
    iz = float(env.scene["object"].data.root_pos_w[0,2].item())
    t = build_target(env, **joints, **RIGHT_HAND_OPEN)
    step_hold_kinematic(env, t, 1)
    robot = env.scene["robot"]
    names = list(robot.data.body_names)
    wrist = robot.data.body_pos_w[0, names.index("right_wrist_yaw_link")]
    palm = robot.data.body_pos_w[0, names.index("right_hand_palm_link")]
    obj = env.scene["object"].data.root_pos_w[0]
    oz = float(obj[2].item())
    out.append({"label": label, "palm_dist": palm_object_distance(env), "object_z": oz, "object_ok": oz > iz-0.02,
                "wrist_z": float(wrist[2]), "palm": [float(x) for x in palm.tolist()], "object": [float(x) for x in obj.tolist()], "joints": joints})
print(json.dumps(out, indent=2))
env.close(); simulation_app.close()
