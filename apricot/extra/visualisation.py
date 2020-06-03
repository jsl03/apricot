"""
Visualisation utilities.
"""
from typing import Optional, Dict, Any
from matplotlib import pyplot as plt  # type: ignore
from matplotlib import gridspec  # type: ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
import numpy as np  # type: ignore
from scipy import interpolate  # type: ignore


def surface(
        x_data: np.ndarray,
        y_data: np.ndarray,
        plot_points: bool = True,
        ax=None,  # type: ignore
        cmap: str = 'plasma',
        resolution: int = 200,
        method: str = 'nearest',
        points_kws: Optional[Dict[str, Any]] = None,
        x1_label: str = 'x1',
        x2_label: str = 'x2',
        y_label: str = 'y',
        aspect: str = 'equal'
):
    """ Surface plot

    Surface plot 2D data.

    Parameters
    ----------
    x_data: ndarray
        (n, 2) array of points
    y_data: ndarray
        (n) array of points
    plot_points: bool, optional
        If True, plot the inidivudal datapoints used to create the surface
        plot. If False, do not plot the points. Default = False.
    ax: matplotlib axis, optional
        If supplied, plot the surface onto the supplied axis. If None, an
        axis will be created. Default None.
    cmap: str, optional
        String representing the desired colormap. Default = 'plasma'.
    resolution: int, optional
        The resolution of the contour plot. Default = 200.
    method: {'nearest', 'linear'}
        String representing the desired interpolation method used to construct
        the contour plot. Default = 'nearest'.
    points_kws: {dict, None}, optional
        If plot_points is True, contains any (optional) additional keyword
        arguments supplied when plotting the datapoints. Default = None.
    x1_label: {str, None}, optional
        String to be used as a label for the x axis. Default = 'x1'.
    x2_label: {str, None}, optional
        String to be used as a label for the y axis. Default = 'x2'.
    y_label: {str, None}, optional
        String to be used as a label for the z axis. Default = 'y'.
    aspect: str, optional
        String representing the desired aspect ratio setting.

    Returns
    -------

    """
    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    if x_data.ndim != 2:
        raise ValueError('x_data must be 2D.')
    x_min = np.min(x_data, axis=0)
    x_max = np.max(x_data, axis=0)
    x_dim_1 = x_data[:, 0]
    x_dim_2 = x_data[:, 1]
    x_grid_1 = np.linspace(x_min[0], x_max[0], resolution)
    x_grid_2 = np.linspace(x_min[1], x_max[1], resolution)
    y_grid = interpolate.griddata(
        (x_dim_1, x_dim_2),
        y_data,
        (x_grid_1[None, :], x_grid_2[:, None]),
        method=method,
        rescale=True
    )
    if ax is None:
        plt.figure(figsize=(5, 5))
        grid = gridspec.GridSpec(1, 1)
        axis = plt.subplot(grid[0], aspect=aspect)
    if points_kws is None:
        points_kws = {}
    points_opts = {
        'linestyle': '',
        'marker': 'x',
        'alpha': 0.75,
        'color': 'w'
    }
    points_opts.update(points_kws)
    contour_plot = ax.contourf(
        x_grid_1,
        x_grid_2,
        y_grid,
        levels=20,
        cmap=cmap
    )
    if plot_points:
        ax.plot(x_dim_1, x_dim_2, **points_opts)
    axis.set_xlabel(x1_label)
    axis.set_ylabel(x2_label)
    div = make_axes_locatable(axis)
    axis_colorbar = div.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(contour_plot, cax=axis_colorbar)
    cbar.set_label(y_label, fontsize=14)
    return axis
