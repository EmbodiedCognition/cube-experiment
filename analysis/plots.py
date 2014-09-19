import contextlib
import lmj.plot


@contextlib.contextmanager
def space(show=True):
    ax = lmj.plot.axes(111, projection='3d', aspect='equal')
    yield ax
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 2])
    if show:
        lmj.plot.show()
