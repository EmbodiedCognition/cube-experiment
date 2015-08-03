import climate
import lmj.cubes
import lmj.plot
import numpy as np


@lmj.cubes.utils.pickled
def jacobian(root, pattern, frames):
    trial = list(lmj.cubes.Experiment(root).trials_matching(pattern))[0]
    trial.load()
    stats = trial.add_jacobian(frames)
    cols = [c for c in trial.df.columns if c.startswith('jac')]
    return stats, trial.df[cols]


def main(root, pattern='68/*block03/*trial00', frames=10, start=27.5, stop=38.5):
    trial = list(lmj.cubes.Experiment(root).trials_matching(pattern))[0]
    trial.load()

    d = trial.distance_to_target[start:stop]
    ax = lmj.plot.create_axes(111, spines=True)
    ax.plot(d, color='#111111', lw=2)
    ax.set_ylabel('Distance to Target (m)')
    ax.set_xlim(0, len(d))
    lmj.plot.show()

    _, jac = jacobian(root, pattern, frames)
    find = lambda b: jac.loc[
        start:stop, [c for c in jac.columns if 'g13x' in c and c.endswith(b)]].T
    kw = dict(vmin=-1, vmax=1, cmap='RdBu')
    lmj.plot.create_axes(111, spines=False).imshow(find('x'), **kw)
    lmj.plot.show()
    return
    lmj.plot.create_axes(413, spines=False).imshow(find('y'), **kw)
    lmj.plot.create_axes(414, spines=False).imshow(find('z'), **kw)


if __name__ == '__main__':
    climate.call(main)
