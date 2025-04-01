import torch
from torch import Tensor
from .tensor_dataclass import TensorDataclass
from .tensor_dataclass_batch_plugins import TensorDataclassBatchPlugin
from jaxtyping import Float, jaxtyped
import typeguard
from functools import reduce


# Define the TestTensorDataclass as provided
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


def test_batch_plugins():
    """
    Test the TensorDataclassBatchPlugin's functionality with TestTensorDataclass.
    """
    # Set B=2, T=1 to ensure compatibility with type hint '*batch 1 10' for betas
    B, T = [2, 4, 5], 10
    test_data = TestTensorDataclass(
        betas=torch.randn(*B, 1, 10),
        joints_wrt_world=torch.randn(*B, T, 24, 3),
        body_quats=torch.randn(*B, T, 21, 4),
        contacts=torch.randint(0, 2, (*B, T, 52)).float(),
    )

    # Define batch dimensions to flatten both B and T
    batch_dims = B

    # **Test 1: Flattening batch dimensions**
    flattened = TensorDataclassBatchPlugin.flatten_obj(test_data, batch_dims)
    assert isinstance(flattened, TestTensorDataclass), (
        "Flattened object should be a TestTensorDataclass"
    )

    cumul_B = reduce(lambda x, y: x * y, batch_dims)
    # Expected shapes after flattening (B*T = 2*1 = 2)
    assert flattened.betas.shape == (cumul_B, 1, 10), (
        f"Expected betas shape {(cumul_B, 1, 10)}, got {flattened.betas.shape}"
    )
    assert flattened.joints_wrt_world.shape == (cumul_B, T, 24, 3), (
        f"Expected joints_wrt_world shape {(cumul_B, T, 24, 3)}"
    )
    assert flattened.body_quats.shape == (cumul_B, T, 21, 4), (
        f"Expected body_quats shape {(cumul_B, T, 21, 4)}"
    )
    assert flattened.contacts.shape == (cumul_B, T, 52), (
        f"Expected contacts shape {(cumul_B, T, 52)}"
    )

    # **Test 2: Unflattening batch dimensions**
    unflattened = TensorDataclassBatchPlugin.unflatten_obj(
        test_data,
        flattened,
        batch_dims,
    )
    assert isinstance(unflattened, TestTensorDataclass), (
        "Unflattened object should be a TestTensorDataclass"
    )

    # Expected shapes after unflattening (should match original shapes since T=1)
    assert unflattened.betas.shape == (*B, 1, 10), f"Expected betas shape {(*B, 1, 10)}"
    assert unflattened.joints_wrt_world.shape == (*B, T, 24, 3), (
        f"Expected joints_wrt_world shape {(*B, T, 24, 3)}"
    )
    assert unflattened.body_quats.shape == (*B, T, 21, 4), (
        f"Expected body_quats shape {(*B, T, 21, 4)}"
    )
    assert unflattened.contacts.shape == (*B, T, 52), (
        f"Expected contacts shape {(*B, T, 52)}"
    )

    # Check values after unflattening
    assert torch.allclose(unflattened.betas, test_data.betas), (
        "Unflattened betas values mismatch"
    )
    assert torch.allclose(unflattened.joints_wrt_world, test_data.joints_wrt_world), (
        "Unflattened joints values mismatch"
    )
    assert torch.allclose(unflattened.body_quats, test_data.body_quats), (
        "Unflattened quats values mismatch"
    )
    assert torch.allclose(unflattened.contacts, test_data.contacts), (
        "Unflattened contacts values mismatch"
    )

    # **Test 3: Apply with flattened batch - Identity function**
    def identity_fn(obj):
        return obj

    result = TensorDataclassBatchPlugin.apply_with_flattened_batch(
        identity_fn,
        test_data,
        batch_dims,
    )
    assert isinstance(result, TestTensorDataclass), (
        "Result with identity_fn should be a TestTensorDataclass"
    )

    # Check shapes and values
    assert result.betas.shape == (*B, 1, 10)
    assert torch.allclose(result.betas, test_data.betas), (
        "Identity function altered betas"
    )
    assert torch.allclose(result.joints_wrt_world, test_data.joints_wrt_world), (
        "Identity function altered joints"
    )
    assert torch.allclose(result.body_quats, test_data.body_quats), (
        "Identity function altered quats"
    )
    assert torch.allclose(result.contacts, test_data.contacts), (
        "Identity function altered contacts"
    )

    # **Test 4: Apply with flattened batch - Doubling function**
    def double_fn(obj):
        return TestTensorDataclass(
            betas=2 * obj.betas,
            joints_wrt_world=2 * obj.joints_wrt_world,
            body_quats=2 * obj.body_quats,
            contacts=2 * obj.contacts,
        )

    result = TensorDataclassBatchPlugin.apply_with_flattened_batch(
        double_fn,
        test_data,
        batch_dims,
    )
    assert isinstance(result, TestTensorDataclass), (
        "Result with double_fn should be a TestTensorDataclass"
    )

    # Check shapes and values
    assert result.betas.shape == (*B, 1, 10)
    assert torch.allclose(result.betas, 2 * test_data.betas), (
        "Doubling function failed for betas"
    )
    assert torch.allclose(result.joints_wrt_world, 2 * test_data.joints_wrt_world), (
        "Doubling function failed for joints"
    )
    assert torch.allclose(result.body_quats, 2 * test_data.body_quats), (
        "Doubling function failed for quats"
    )
    assert torch.allclose(result.contacts, 2 * test_data.contacts), (
        "Doubling function failed for contacts"
    )

    # **Test 5: Flattening with batch_dims=[B] only**
    batch_dims = B[:-1]
    cumul_B = reduce(lambda x, y: x * y, batch_dims)
    flattened = TensorDataclassBatchPlugin.flatten_obj(test_data, batch_dims)

    # Expected shapes (only B is flattened, T remains)
    assert flattened.betas.shape == (cumul_B, B[-1], 1, 10), (
        f"Expected betas shape {(cumul_B, B[-1], 1, 10)}"
    )
    assert flattened.joints_wrt_world.shape == (cumul_B, B[-1], T, 24, 3), (
        f"Expected joints_wrt_world shape {(cumul_B, B[-1], T, 24, 3)}"
    )
    assert flattened.body_quats.shape == (cumul_B, B[-1], T, 21, 4), (
        f"Expected body_quats shape {(cumul_B, B[-1], T, 21, 4)}"
    )
    assert flattened.contacts.shape == (cumul_B, B[-1], T, 52), (
        f"Expected contacts shape {(cumul_B, B[-1], T, 52)}"
    )

    # Apply doubling function with batch_dims=[B]
    result = TensorDataclassBatchPlugin.apply_with_flattened_batch(
        double_fn,
        test_data,
        batch_dims,
    )
    assert result.betas.shape == (*B, 1, 10)
    assert torch.allclose(result.betas, 2 * test_data.betas), (
        "Doubling with batch_dims=[B] failed for betas"
    )
    assert torch.allclose(result.joints_wrt_world, 2 * test_data.joints_wrt_world), (
        "Doubling with batch_dims=[B] failed for joints"
    )


if __name__ == "__main__":
    test_batch_plugins()
    print("All tests passed!")
