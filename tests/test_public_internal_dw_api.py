import torch
from torch import nn

from internal_dw import InternalDW, InternalDWResidual


def test_public_router_is_forward_identical_and_route_specific_backward():
    router = InternalDW(
        state_dim=1,
        num_layers=1,
        max_horizon=1,
        warmup_batches=100,
    )
    with torch.no_grad():
        router.gains[0, 0] = torch.tensor([0.25, 0.50])

    weight = nn.Parameter(torch.tensor(2.0))
    x = torch.tensor([3.0], requires_grad=True)
    router.begin_batch()
    skip_x, branch_x = router.route(x, horizon=0, layer=0)
    y = skip_x + weight * branch_x

    assert torch.equal(skip_x, x)
    assert torch.equal(branch_x, x)
    assert torch.equal(y, torch.tensor([9.0]))

    y.sum().backward()
    router.end_batch()
    assert torch.allclose(x.grad, torch.tensor([1.25]))
    # Local branch-parameter gradients remain fully open.
    assert torch.allclose(weight.grad, torch.tensor(3.0))


def test_residual_wrapper_and_state_dict_round_trip():
    router = InternalDW(state_dim=2, num_layers=1, max_horizon=2)
    block = InternalDWResidual(nn.Identity(), router, layer=0)
    x = torch.randn(3, 2)
    with torch.no_grad():
        assert torch.equal(block(x, 0), 2.0 * x)

    clone = InternalDW(state_dim=2, num_layers=1, max_horizon=2)
    clone.load_state_dict(router.state_dict())
    assert torch.equal(clone.gains, router.gains)

