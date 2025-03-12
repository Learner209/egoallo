import pytest
import torch
import tempfile
import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import train_motion_prior
import subprocess
from egoallo.fncsmpl import SmplhModel
import os


@pytest.fixture
def mock_accelerator():
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.device = torch.device("cpu")
    accelerator.prepare = MagicMock(
        return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock()),
    )
    accelerator.register_for_checkpointing = MagicMock()
    accelerator.load_state = MagicMock()
    accelerator.save_state = MagicMock()
    return accelerator


@pytest.fixture
def mock_wandb():
    with patch("wandb.init") as mock_wandb_init:
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_run.log = MagicMock()
        mock_run.finish = MagicMock()
        yield mock_run


@pytest.fixture
def temp_dataset(tmp_path: Path = Path("./assets/dummy_test/data/amass_rich_hps")):
    hdf5_path = tmp_path / "processed_amass_rich_hps_correct.hdf5"
    txt_path = tmp_path / "processed_amass_rich_hps_correct.txt"

    return hdf5_path, txt_path


@pytest.fixture
def mock_body_model():
    model_path = Path("./assets/dummy_test/smplh_model/male/model.npz")
    return SmplhModel.load(model_path)


def test_run_training_real_data(tmp_path, mock_accelerator, mock_wandb, temp_dataset):
    # Use the provided command-line arguments to create a MockConfig
    from egoallo.config.train.train_config import EgoAlloTrainConfig

    config = EgoAlloTrainConfig(
        batch_size=64,
        experiment_name=f"{datetime.datetime.now().strftime('%Y%m%d')}-pytest-run",
        learning_rate=1e-4,
        dataset_hdf5_path=temp_dataset[0],
        dataset_files_path=temp_dataset[1],
        spatial_mask_ratio=0.75,
        splits=("train", "val"),
        joint_cond_mode="absrel",
        use_fourier_in_masked_joints=True,
        random_sample_mask_ratio=True,
        data_collate_fn="TensorOnlyDataclassBatchCollator",
        dataset_type="AdaptiveAmassHdf5Dataset",
        subseq_len=128,
        debug=False,
        max_steps=40,
    )

    # Patch dependencies
    # with patch("train_motion_prior.Accelerator", return_value=mock_accelerator), \
    with patch("wandb.init"), patch("wandb.log"), patch("wandb.finish"):
        # patch("train_motion_prior.accelerator.is_main_process", return_value=True):
        # Run the training function with max_steps
        assert mock_accelerator.is_main_process
        train_motion_prior.run_training(config, debug_mode=False)

        # Assertions
        # assert mock_accelerator.prepare.call_count == 1
        # assert mock_accelerator.save_state.call_count >= 1
        # assert mock_wandb.log.call_count >= 0
        # assert mock_wandb.finish.call_count == 1

        # Check if the experiment directory is created
        experiment_dir = tmp_path / config.experiment_name
        assert experiment_dir.exists()


# Test test.py
def test_eval_on_amass(tmp_path):
    from egoallo.config.inference.inference_defaults import InferenceConfig

    config = InferenceConfig(
        checkpoint_dir=Path(
            "/export/home/chenqixuan/code/muscle/egoallo/experiments/Jan_16_mr_92/v5/checkpoints_150000",
        ),
        output_dir=Path("./exp/test-egoexo-train"),
        device="cpu",
        visualize_traj=True,
        debug_max_iters=5,
        skip_eval_confirm=True,
        dataset_slice_strategy="full_sequence",
        splits=("test",),
        dataset_type="AriaDataset",
        guidance_mode="no_hands",
        egoexo=InferenceConfig.EgoExoConfig(
            split="train",
            use_pseudo=False,
            coord="global",
        ),
    )

    with patch("torch.load"), patch("torch.save"):
        from egoallo.scripts.test import main

        main(config, debug=False)


# Test export_hdf5.py
@pytest.mark.parametrize(
    "export_hdf5_cli_params",
    [
        {
            "data_npz_dirs": [
                Path("./data/my_amass/processed"),
                Path("./data/rich/processed/"),
                Path("./data/hps/processed/"),
            ],
            "output_file": Path(
                "./data/amass_rich_hps/processed_amass_rich_hps_correct.hdf5",
            ),
            "output_list": Path(
                "./data/amass_rich_hps/processed_amass_rich_hps_correct.txt",
            ),
        },
    ],
)
def test_export_hdf5(tmp_path, mock_body_model, test_params):
    from egoallo.scripts.export_hdf5 import main

    with (
        patch("fncsmpl.SmplhModel.load", return_value=mock_body_model),
        patch("torch.cuda.is_available", return_value=True),
    ):
        main(
            data_npz_dirs=test_params["data_npz_dirs"],
            output_file=test_params["output_file"],
            output_list_file=test_params["output_list"],
        )

        # Verify hdf5 created
        assert test_params["output_file"].exists()


# Test visualize_inference.py
# def test_visualize_inference(tmp_path):
#     from egoallo.scripts.visualize_inference import main

#     # Create test trajectory files
#     gt_traj = tmp_path / "gt.pt"
#     est_traj = tmp_path / "est.pt"
#     torch.save({"test": torch.randn(10,3)}, gt_traj)
#     torch.save({"test": torch.randn(10,3)}, est_traj)

#     with patch("torch.load"), \
#          patch("egoallo.scripts.visualize_inference.render_trajectory"):

#         main(
#             trajectory_paths=[gt_traj, est_traj],
#             output_dir=tmp_path,
#             smplh_model_path=tmp_path / "model.npz"
#         )

#         # Verify visualization output
#         assert len(list(tmp_path.glob("*.mp4"))) > 0


def run_cli_command(command):
    """Run a CLI command and return the result."""
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result


@pytest.fixture
def temp_input_file():
    """Create a temporary input file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write("test data")
    yield f.name  # Provide the file path to the test
    os.remove(f.name)  # Clean up after the test


@pytest.mark.parametrize(
    "eval_cli",
    [
        [
            "python",
            "./src/egoallo/scripts/test.py",
            "--inference-config.dataset-slice-strategy",
            "full_sequence",
            "--inference-config.splits",
            "test",
            "--inference-config.checkpoint-dir",
            "/export/home/chenqixuan/code/muscle/egoallo/experiments/combined_2_Jan_mr_0.75_foot_skating_loss/v0/checkpoints_690000",
            "--inference-config.dataset-type",
            "AdaptiveAmassHdf5Dataset",
            "--inference-config.visualize-traj",
            "--inference-config.debug-max-iters",
            "15",
        ],  # Full command test case
        [
            "python",
            "./src/egoallo/scripts/test.py",
            "--inference-config.dataset-slice-strategy",
            "full_sequence",
            "--inference-config.splits",
            "test",
            "--inference-config.checkpoint-dir",
            "/export/home/chenqixuan/code/muscle/egoallo/experiments/combined_2_Jan_mr_0.75_foot_skating_loss/v0/checkpoints_690000",
            "--inference-config.dataset-type",
            "AdaptiveAmassHdf5Dataset",
            "--inference-config.visualize-traj",
            "--inference-config.debug-max-iters",
            "15",
            "--inference-config.guidance-inner",
            "--inference-config.guidance-post",
        ],  # Full command test case
        [
            "python",
            "./src/egoallo/scripts/test.py",
            "--inference-config.dataset-slice-strategy",
            "full_sequence",
            "--inference-config.splits",
            "test",
            "--inference-config.checkpoint-dir",
            "/export/home/chenqixuan/code/muscle/egoallo/experiments/combined_2_Jan_mr_0.75_foot_skating_loss/v0/checkpoints_690000",
            "--inference-config.dataset-type",
            "AriaDataset",
            "--inference-config.visualize-traj",
            "--inference-config.output-dir",
            "./exp/test-egoexo-train",
            "--inference-config.egoexo.split",
            "train",
            "--inference-config.guidance-mode",
            "no_hands",
            "--inference-config.debug-max-iters",
            "5",
            "--inference-config.guidance-inner",
            "--inference-config.guidance-post",
        ],  # Full command test case
        [
            "python",
            "./src/egoallo/scripts/test.py",
            "--inference-config.dataset-slice-strategy",
            "full_sequence",
            "--inference-config.splits",
            "test",
            "--inference-config.checkpoint-dir",
            "/export/home/chenqixuan/code/muscle/egoallo/experiments/combined_2_Jan_mr_0.75_foot_skating_loss/v0/checkpoints_690000",
            "--inference-config.dataset-type",
            "AriaDataset",
            "--inference-config.visualize-traj",
            "--inference-config.output-dir",
            "./exp/test-egoexo-train",
            "--inference-config.egoexo.split",
            "train",
            "--inference-config.guidance-mode",
            "no_hands",
            "--inference-config.debug-max-iters",
            "5",
        ],  # Full command test case
        ["python", "./src/egoallo/scripts/test.py", "--help"],  # Help test case
    ],
)
def test_eval(command):
    """Test that a CLI command runs without errors."""
    result = run_cli_command(command)

    # Assert that the command completed successfully (return code 0)
    assert result.returncode == 0, f"Command failed with error: {result.stderr}"

    # Optionally, check the output or error messages
    if "--help" in command:
        assert "usage:" in result.stdout
    elif "--invalid-arg" in command:
        assert "error:" in result.stderr


def test_cli_with_input_file(temp_input_file):
    """Test a CLI command that reads from a file."""
    command = ["my_script.py", "--input", temp_input_file]
    result = run_cli_command(command)

    assert result.returncode == 0
    assert "processed data" in result.stdout


@pytest.mark.slow
def test_cli_long_running_command():
    """Test a long-running CLI command."""
    command = ["my_script.py", "--long-running"]
    result = run_cli_command(command)

    assert result.returncode == 0
