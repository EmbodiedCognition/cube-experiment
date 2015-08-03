import climate
import lmj.cubes
import lmj.plot
import lmj.pca
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.decomposition


@lmj.cubes.utils.pickled
def jacobian(root, pattern, frames):
    trial = list(lmj.cubes.Experiment(root).trials_matching(pattern))[0]
    trial.load()
    stats = trial.add_jacobian(frames)
    return stats, trial.df[[c for c in trial.df.columns if c.startswith('jac')]]

PCA_FILE = '/tmp/jacobian.npz'
MODEL = '/tmp/sparse-coder-100-0.0001.pkl'

def main(root, pattern='68/*block03/*trial00', frames=10, start=27.5, stop=38.5):
    _, jac = jacobian(root, pattern, frames)
    n = int(np.sqrt(len(jac.columns) / 9))

    pca = lmj.pca.PCA(filename=PCA_FILE)
    for v in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999):
        print('variance', v, 'components', pca.num_components(v))

    if os.path.exists(MODEL):
        dl = pickle.load(open(MODEL, 'rb'))
    else:
        dl = sklearn.decomposition.MiniBatchSparsePCA(
            n_components=100, alpha=0.0001, verbose=10)
        dl.fit(pca.encode(jac.dropna().values, retain=0.999))
        pickle.dump(dl, open(MODEL, 'wb'), -1)

    df = pd.DataFrame(pca.decode(dl.components_, retain=0.999), columns=jac.columns)

    enc = sklearn.decomposition.sparse_encode(
        pca.encode(jac.dropna().loc[start:stop, :], retain=0.999),
        dl.components_,
        algorithm='omp',
        alpha=0.0001,
    )

    kw = dict(vmin=-10, vmax=10, cmap='RdBu')
    lmj.plot.create_axes(111, spines=False).imshow(enc.T, **kw)
    lmj.plot.show()

    enc = sklearn.decomposition.sparse_encode(
        jac.dropna().loc[start:stop, :],
        pca.vecs.T[:len(dl.components_)],
        algorithm='omp',
        alpha=0.0001,
    )

    kw = dict(vmin=-10, vmax=10, cmap='RdBu')
    lmj.plot.create_axes(111, spines=False).imshow(enc.T, **kw)
    lmj.plot.show()

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
