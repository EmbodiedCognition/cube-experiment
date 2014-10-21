import contextlib
import lmj.plot
import numpy as np


@contextlib.contextmanager
def plot(show=True):
    ax = lmj.plot.axes()
    yield ax
    if show:
        lmj.plot.show()


@contextlib.contextmanager
def space(show=True):
    ax = lmj.plot.axes(111, projection='3d', aspect='equal')
    yield ax
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 2])
    if show:
        lmj.plot.show()


u, v = np.mgrid[0:2 * np.pi:11j, 0:np.pi:7j]
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
