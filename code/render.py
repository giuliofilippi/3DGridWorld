# imports
import numpy as np
from mayavi import mlab

# render tensor with mayavi
def render(tensor, show=True, save=False, name="image.png"):
    """
    Render a tensor using Mayavi for visualization.

    Parameters:
    - tensor: The tensor object.
    - show: Whether to display the visualization.
    - save: Whether to save the visualization.
    - name: Name of the saved image.
    """
    # figure
    fig = mlab.figure(size=(1024, 1024))
    # Define colors based on your values
    render_list = {
        1: (1, 1, 1),    # Soil
        -1: (1, 0, 0),    # Objects
        -2:(0, 0, 1),    # Agents
        2:(1, 0.75, 0) # Built structure
        }

    # plotting voxels
    for value, color in render_list.items():
        x_vals, y_vals, z_vals = np.where(tensor == value)
        mlab.points3d(x_vals, y_vals, z_vals,
                        mode="cube",
                        color=color,
                        scale_factor=1)

    # save image
    if save:
        mlab.savefig(name)
        mlab.close()
    # show image
    if show:
        mlab.show()