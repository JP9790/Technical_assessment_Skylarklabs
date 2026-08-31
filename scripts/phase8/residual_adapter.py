"""Small residual MLP: q_out = q_base + adapter(q_base)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

JP_TEST_ROOT = Path(__file__).resolve().parents[2]


class ResidualAdapter(nn.Module if nn else object):
    def __init__(self, in_dim: int = 6, hidden: int = 64, out_dim: int = 6) -> None:
        if nn is None:
            raise RuntimeError("PyTorch required for ResidualAdapter")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, q_base: torch.Tensor) -> torch.Tensor:
        return self.net(q_base)

    def predict(self, q_base: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(q_base, dtype=torch.float32)
            if next(self.parameters()).is_cuda:
                t = t.cuda()
            out = self(t)
            return out.cpu().numpy()


def save_adapter(model: ResidualAdapter, path: Path, *, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta or {}}
    torch.save(payload, path)


def load_adapter(path: Path, *, hidden: int = 64) -> ResidualAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    in_dim = int(payload.get("meta", {}).get("in_dim", 6))
    out_dim = int(payload.get("meta", {}).get("out_dim", 6))
    hidden = int(payload.get("meta", {}).get("hidden", hidden))
    model = ResidualAdapter(in_dim, hidden, out_dim)
    model.load_state_dict(payload["state_dict"])
    return model
