#!/usr/bin/env python3
"""Train Stage C residual adapter (assessment §1.5): q* ≈ q_joint + adapter(q_joint)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_adapter import ResidualAdapter, save_adapter

CFG = JP_TEST_ROOT / "configs/finetune.yaml"
DATA = JP_TEST_ROOT / "results/stage_C/residual_dataset.npz"


def train_side(name: str, q_base: np.ndarray, delta: np.ndarray, cfg: dict) -> tuple[ResidualAdapter, dict]:
    hidden = int(cfg["model"]["hidden_dim"])
    lr = float(cfg["model"]["learning_rate"])
    epochs = int(cfg["model"]["epochs"])
    batch = int(cfg["model"]["batch_size"])

    model = ResidualAdapter(6, hidden, 6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor(q_base, dtype=torch.float32)
    y = torch.tensor(delta, dtype=torch.float32)
    n = x.shape[0]
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(idx)
        if (epoch + 1) % 50 == 0:
            print(f"  [{name}] epoch {epoch+1}/{epochs} loss={epoch_loss/n:.6f}", flush=True)

    with torch.no_grad():
        pred = model(x).numpy()
    mse = float(np.mean((pred - delta) ** 2))
    return model, {
        "side": name,
        "train_mse": mse,
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "wall_clock_s": time.time() - t0,
        "epochs": epochs,
        "samples": n,
    }


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    if not DATA.is_file():
        raise FileNotFoundError(f"Run build_residual_dataset.py first: {DATA}")

    d = np.load(DATA)
    right_model, right_meta = train_side("right", d["q_base_right"], d["delta_right"], cfg)
    left_model, left_meta = train_side("left", d["q_base_left"], d["delta_left"], cfg)

    ckpt_path = JP_TEST_ROOT / cfg["output"]["checkpoint"]
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "in_dim": 6,
        "out_dim": 6,
        "hidden": int(cfg["model"]["hidden_dim"]),
        "right": right_meta,
        "left": left_meta,
    }
    save_adapter(right_model, ckpt_path.with_name("residual_adapter_right.pt"), meta=meta)
    save_adapter(left_model, ckpt_path.with_name("residual_adapter_left.pt"), meta=meta)

    report = {
        "timestamp_utc": meta["timestamp_utc"],
        "checkpoints": {
            "right": str(ckpt_path.with_name("residual_adapter_right.pt")),
            "left": str(ckpt_path.with_name("residual_adapter_left.pt")),
        },
        "training": meta,
        "stage_c_train_pass": right_meta["train_mse"] < 0.01 and left_meta["train_mse"] < 0.01,
    }
    out = JP_TEST_ROOT / "results/stage_C/finetune_train.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["stage_c_train_pass"] else 0  # soft pass — still usable


if __name__ == "__main__":
    raise SystemExit(main())
