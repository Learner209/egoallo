import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_env_vars():
    # Set environment variables from your local shell
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/lib/nvidia:/usr/local/cuda-12.6/lib64:/usr/lib/nvidia:/usr/local/cuda-12.6/lib64::/home/minghao/.mujoco/mujoco210/bin:/home/minghao/.mujoco/mujoco210/bin "
    )
    os.environ["PATH"] = (
        "/home/minghao/.autojump/bin:/home/minghao/.autojump/bin:/usr/local/cuda-12.6/bin:/home/minghao/.local/bin:/home/minghao/.autojump/bin:/home/minghao/.autojump/bin:/home/minghao/.nvm/versions/node/v22.12.0/bin:/usr/local/cuda-12.6/bin:/home/minghao/miniconda3/envs/egoallo/bin:/home/minghao/miniconda3/condabin:/home/minghao/.local/bin:/home/minghao/.local/bin:/home/minghao/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/minghao/.local/share/JetBrains/Toolbox/scripts:/home/minghao/opt/bin:/usr/sbin:/usr/bin:/usr/local/go/bin:/home/minghao/.fzf/bin:/home/minghao/opt/bin:/usr/sbin:/usr/bin:/usr/local/go/bin"
    )
