import torch
from torch import Tensor
from .tensor_dataclass import TensorDataclass
from jaxtyping import Float, jaxtyped
import typeguard


@jaxtyped(typechecker=typeguard.typechecked)
class TestTensorDataclass(TensorDataclass):
    contacts: Float[Tensor, "*batch timesteps 52"]
    """Contact boolean for each joint."""

    betas: Float[Tensor, "*batch 1 10"]
    """Body shape parameters. Default to 10 when using smplx model."""

    joints_wrt_world: Float[Tensor, "*batch timesteps 24 3"]
    """Joint positions relative to the world frame."""

    body_quats: Float[Tensor, "*batch timesteps 21 4"]
    """Local orientations for each body joint."""


def test_slicing_behavior():
    # Create test data
    B, T = 2, 5
    test_data = TestTensorDataclass(
        betas=torch.randn(B, 1, 10),
        joints_wrt_world=torch.randn(B, T, 24, 3),
        body_quats=torch.randn(B, T, 21, 4),
        contacts=torch.randint(0, 2, (B, T, 52)).float(),
    )

    # Test 1: Basic slicing
    sliced = test_data[:1]
    assert sliced.betas.shape == (1, 1, 10)
    assert sliced.joints_wrt_world.shape == (1, T, 24, 3)
    assert sliced.body_quats.shape == (1, T, 21, 4)
    assert sliced.contacts.shape == (1, T, 52)

    # Test 2: Out-of-bound slice on size-1 dimension
    sliced = test_data[:, 3:55]  # Should preserve size-1 dim
    assert sliced.betas.shape == (B, 1, 10)  # Dim preserved
    assert sliced.joints_wrt_world.shape == (B, 2, 24, 3)  # 5-3=2 (original T=5)
    assert sliced.contacts.shape == (B, 2, 52)

    # Test 3: Completely out-of-bound slice
    sliced = test_data[:, 100:200]
    assert sliced.betas.shape == (B, 1, 10)  # Dim preserved
    assert sliced.joints_wrt_world.shape == (B, 0, 24, 3)  # Empty slice
    assert sliced.contacts.shape == (B, 0, 52)

    # Test 5: Optional tensor handling
    sliced = test_data[:, :1]
    assert sliced.betas.shape == (B, 1, 10)
    assert sliced.joints_wrt_world.shape == (B, 1, 24, 3)
    assert sliced.body_quats.shape == (B, 1, 21, 4)
    assert sliced.contacts.shape == (B, 1, 52)

    # Test 6: Wrong tensor handling
    try:
        sliced = test_data[:, :, :, 4:5]
    except IndexError as e:
        print(f"Caught expected IndexError: {e}")
    else:
        raise AssertionError("Expected IndexError was not raised")

    # Test 8: negative slice
    sliced = test_data[:, :-1]
    assert sliced.betas.shape == (B, 1, 10)
    assert sliced.joints_wrt_world.shape == (B, T - 1, 24, 3)
    assert sliced.body_quats.shape == (B, T - 1, 21, 4)
    assert sliced.contacts.shape == (B, T - 1, 52)


if __name__ == "__main__":
    test_slicing_behavior()
