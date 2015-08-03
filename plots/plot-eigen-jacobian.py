import climate
import lmj.cubes
import lmj.plot
import lmj.pca
import numpy as np
import os
import pandas as pd


@lmj.cubes.utils.pickled
def jacobian(root, pattern, frames):
    trial = list(lmj.cubes.Experiment(root).trials_matching(pattern))[0]
    trial.load()
    stats = trial.add_jacobian(frames)
    return stats, trial.df[[c for c in trial.df.columns if c.startswith('jac')]]

PCA_FILE = '/tmp/jacobian.npz'


def main(root, pattern='68/*block03/*trial00', frames=10):
    _, jac = jacobian(root, pattern, frames)
    n = int(np.sqrt(len(jac.columns) / 9))

    pca = lmj.pca.PCA()
    if os.path.exists(PCA_FILE):
        pca.load(PCA_FILE)
    else:
        pca.fit(jac.dropna().values.astype('f'))
        print('saving', PCA_FILE)
        pca.save(PCA_FILE)

    df = pd.DataFrame(pca.vecs.T, columns=jac.columns)

    def find(i, g, b):
        cs = [c for c in jac.columns if '{}/'.format(g) in c and c.endswith(b)]
        return df[cs].loc[i, :].values.reshape((n, n))

    def plot(where, i, g, b):
        ax = lmj.plot.create_axes(where, spines=False)
        ax.imshow(find(i, g, b), cmap='RdBu')

    for i in range(10):
        plot(331, i, 'x', 'x')
        plot(332, i, 'x', 'y')
        plot(333, i, 'x', 'z')
        plot(334, i, 'y', 'x')
        plot(335, i, 'y', 'y')
        plot(336, i, 'y', 'z')
        plot(337, i, 'z', 'x')
        plot(338, i, 'z', 'y')
        plot(339, i, 'z', 'z')
        lmj.plot.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
