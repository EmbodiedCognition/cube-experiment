import contextlib
import lmj.plot
import numpy as np


@contextlib.contextmanager
def plot(show=True):
    '''Produce a 2D plotting axes, for use in a with statement.'''
    ax = lmj.plot.axes()
    yield ax
    if show:
        lmj.plot.show()


@contextlib.contextmanager
def space(show=True):
    '''Produce a 3D plotting axes, for use in a with statement.'''
    ax = lmj.plot.axes(111, projection='3d', aspect='equal')
    yield ax
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([ 0, 2])
    ax.w_xaxis.set_pane_color((1, 1, 1, 1))
    ax.w_yaxis.set_pane_color((1, 1, 1, 1))
    ax.w_zaxis.set_pane_color((1, 1, 1, 1))
    if show:
        lmj.plot.show()


def show_cubes(ax, trial, target_num=None):
    '''Plot markers for target cubes on the given axes.

    Parameters
    ----------
    ax : plotting axes
    trial : Trial object
    target_num : int or set of int, optional
        If given, only show this/these targets.
    '''
    if target_num is None:
        target_num = range(12)
    elif isinstance(target_num, int):
        target_num = (target_num, )
    to_plot = set(target_num)
    xs, ys, zs = [], [], []
    for flavor in ('source', 'target'):
        cols = [flavor + suffix for suffix in ('', '-x', '-y', '-z')]
        for _, (n, x, y, z) in trial.df[cols].drop_duplicates().iterrows():
            if n in to_plot:
                to_plot.remove(n)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                ax.text(x, z + 0.1, y + 0.1, str(int(n)))
    ax.scatter(xs, zs, ys, marker='o', s=200, color='#111111', linewidth=0, alpha=0.7)


def skeleton(ax, trial, frame):
    '''Plot a skeleton based on a frame of the given trial.

    Parameters
    ----------
    ax : Axes
    trial : Movement
    frame : int or float
    '''
    for segment in (
        # right leg
        [34, 44, 45, 46], #47, 48, 49],
        # left leg
        [35, 37, 38, 39], #40, 41, 42],
        # right arm
        [32, 6, 7, 8, 9, 15],
        # left arm
        [32, 18, 19, 20, 21, 27],
        # right hand
        [10, 14, 11, 14, 12, 15, 13, 15, 16, 17],
        # left hand
        [26, 22, 26, 23, 27, 24, 27, 25, 28, 29],
        # head + torso
        [1, 4, 5, 3, 2, 0, 32, 33, 34, 35, 36, 43, 30, 31],
    ):
        xs, ys, zs = [], [], []
        for marker in segment:
            traj = trial.trajectory(marker)
            xs.append(traj.x[frame])
            ys.append(traj.y[frame])
            zs.append(traj.z[frame])
        ax.plot(xs, zs, zs=ys, **kwargs)


u, v = np.mgrid[0:2 * np.pi:17j, 0:np.pi:13j]
sphere = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

def ellipsoid(center, radius):
    '''Return a grid of points defining an ellipsoid at the given location.

    At the moment this function returns an axis-aligned ellipsoid.

    Parameters
    ----------
    center : array (3, )
        The center of the ellipsoid.
    radius : array (3, )
        The radius of the ellipsoid along each of the coordinate axes.

    Returns
    -------
    array (3, lat, long) :
        An array of points on the surface of an axis-aligned ellipsoid.
    '''
    return (center + sphere.T * radius).T
