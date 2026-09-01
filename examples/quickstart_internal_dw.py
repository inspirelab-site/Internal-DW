"""Minimal Internal-DW integration in a residual forecaster.

Run with:
    python examples/quickstart_internal_dw.py
"""

import torch
from torch import nn

from internal_dw import InternalDW, InternalDWResidual


class TinyForecaster(nn.Module):
    def __init__(self, width: int, horizon: int) -> None:
        super().__init__()
        self.dw = InternalDW(
            state_dim=width,
            num_layers=2,
            max_horizon=horizon,
            warmup_batches=1,
            probe_every=1,
        )
        self.blocks = nn.ModuleList(
            [
                InternalDWResidual(
                    nn.Sequential(nn.Linear(width, width), nn.Tanh()),
                    self.dw,
                    layer=layer,
                )
                for layer in range(2)
            ]
        )

    def step(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, horizon)
        return x


def main() -> None:
    torch.manual_seed(0)
    model = TinyForecaster(width=8, horizon=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for batch in range(3):
        history = torch.randn(16, 8)
        targets = torch.randn(16, 4, 8)

        optimizer.zero_grad(set_to_none=True)
        model.dw.begin_batch()
        state = history
        losses = []
        for k in range(4):
            state = model.step(state, horizon=k)
            losses.append((state - targets[:, k]).square().mean())
            model.dw.observe(state, targets[:, k], horizon=k)

        loss = torch.stack(losses).mean()
        model.dw.calibrate()  # read-only route VJP probes
        loss.backward()       # optimizer gradient with the current gains
        model.dw.end_batch()  # gains measured above take effect next batch
        optimizer.step()
        print(
            f"batch={batch} loss={loss.item():.4f} "
            f"mean_gain={model.dw.gains.mean().item():.3f}"
        )

    print(model.dw.gain(horizon=0, layer=0))


if __name__ == "__main__":
    main()

