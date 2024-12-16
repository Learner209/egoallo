import matplotlib.pyplot as plt
import numpy as np
import torch  # Assuming the error is a PyTorch tensor

from egoallo.utils.smpl_mapping.mapping import EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS, EGOEXO4D_EGOPOSE_HANDPOSE_MAPPINGS

BODY_JOINTS = EGOEXO4D_EGOPOSE_BODYPOSE_MAPPINGS
NUM_OF_BODY_JOINTS = len(BODY_JOINTS)  

def draw_err(error):
    """
    error: (T, J, 3)
    """
    error = np.squeeze(error)  # Remove the first dimension if it's size 1
    error_x = error[:, :, 0]  # Shape: (500, 17)
    error_y = error[:, :, 1]  # Shape: (500, 17)
    error_z = error[:, :, 2]  # Shape: (500, 17)
    error_overall = np.sqrt(error_x**2 + error_y**2 + error_z**2)  # Euclidean norm for overall error

    # Create a figure and a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Titles for each subplot
    titles = ['X-Axis Error', 'Y-Axis Error', 'Z-Axis Error', 'Overall Error']
    data = [error_x, error_y, error_z, error_overall]
    time_steps = np.arange(error.shape[0])

    # Plot each axis in a subplot
    for ax, title, axis_data in zip(axes.flat, titles, data):
        for j in range(axis_data.shape[1]):  # loop over each joint
            ax.plot(time_steps, axis_data[:, j], label=f'{BODY_JOINTS[j]}')
        ax.set_title(title)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Error')
        ax.grid(True)
        # Optionally add a legend if you want to identify the lines
        ax.legend()

    # Adjust layout to not overlap
    plt.tight_layout()
    # plt.show()

    return fig
