import climate
import lmj.cubes
import lmj.plot
import numpy as np


@lmj.cubes.utils.pickled
def jacobian(root, pattern, frames):
    trial = list(lmj.cubes.Experiment(root).trials_matching(pattern))[0]
    trial.load()
    return trial.jacobian(frames)


def main(root, pattern='68/*block03/*trial00', frames=10, frame=29.7): # 34.5
    _, jac = jacobian(root, pattern, frames)
    cols = jac.columns
    n = int(np.sqrt(len(cols) / 9))
    def find(g, b):
        cs = [c for c in cols if '{}/'.format(g) in c and c.endswith(b)]
        return jac[cs].loc[frame, :].values.reshape((n, n))
    def plot(where, g, b):
        ax = lmj.plot.create_axes(where, spines=False)
        ax.colorbar(ax.imshow(find(g, b), vmin=-1, vmax=1, cmap='coolwarm'))
        #ax.set_xlabel('{}{}'.format(g, b))
    plot(331, 'x', 'x')
    plot(332, 'x', 'y')
    plot(333, 'x', 'z')
    plot(334, 'y', 'x')
    plot(335, 'y', 'y')
    plot(336, 'y', 'z')
    plot(337, 'z', 'x')
    plot(338, 'z', 'y')
    plot(339, 'z', 'z')
    lmj.plot.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
