import contextlib
import lmj.plot
import numpy as np


@contextlib.contextmanager
def space(show_afterwards=True):
    '''Produce a 3D plotting axes, for use in a with statement.'''
    ax = lmj.plot.axes(111, projection='3d', aspect='equal')
    yield ax
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([ 0, 2])
    ax.w_xaxis.set_pane_color((1, 1, 1, 1))
    ax.w_yaxis.set_pane_color((1, 1, 1, 1))
    ax.w_zaxis.set_pane_color((1, 1, 1, 1))
    if show_afterwards:
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


def skeleton(ax, trial, frame, **kwargs):
    '''Plot a skeleton based on a frame of the given trial.

    Parameters
    ----------
    ax : Axes
    trial : Movement
    frame : int or float
    '''
    for segment in (
        # legs
        ['l-ilium', 'l-knee', 'l-shin', 'l-ankle', 'l-heel', 'l-mt-outer', 'l-mt-inner'],
        ['r-ilium', 'r-knee', 'r-shin', 'r-ankle', 'r-heel', 'r-mt-outer', 'r-mt-inner'],
        # arms
        ['t3', 'r-collar', 'r-shoulder', 'r-elbow', 'r-wrist', 'r-mc-inner'],
        ['t3', 'l-collar', 'l-shoulder', 'l-elbow', 'l-wrist', 'l-mc-inner'],
        # hands
        ['r-fing-pinky', 'r-mc-outer', 'r-fing-ring', 'r-mc-outer', 'r-fing-middle',
         'r-mc-inner', 'r-fing-index', 'r-mc-inner', 'r-thumb-base', 'r-thumb-tip'],
        ['l-fing-pinky', 'l-mc-outer', 'l-fing-ring', 'l-mc-outer', 'l-fing-middle',
         'l-mc-inner', 'l-fing-index', 'l-mc-inner', 'l-thumb-base', 'l-thumb-tip'],
        # head + torso
        ['r-head-front', 'r-head-mid', 'l-head-mid', 'l-head-back', 'l-head-front', 'r-head-back',
         't3', 't9', 'l-ilium', 'r-ilium', 'r-hip', 'l-hip', 'abdomen', 'sternum'],
    ):
        xs, ys, zs = [], [], []
        for marker in segment:
            traj = trial.trajectory(marker)
            xs.append(traj.x[traj.index[frame]])
            ys.append(traj.y[traj.index[frame]])
            zs.append(traj.z[traj.index[frame]])
        ax.scatter(xs, zs, zs=ys, s=10, c='#111111', alpha=0.5, **kwargs)
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
