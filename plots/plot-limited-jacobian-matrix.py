import climate
import lmj.cubes
import lmj.plot as plt
import numpy as np

MARKERS = [2, 13, 25, 31, 33, 40, 47]


def main(root, pattern='*'):
    bins = np.logspace(-4, 2, 127)

    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        stats, jac = t.jacobian(10, markers=MARKERS)

        plt.colorbar(plt.imshow(jac.values[500:800].T,
                                vmin=-1, vmax=1, cmap='coolwarm'))
        plt.show()

        hist = np.zeros((len(jac), len(bins)), float)
        for t, j in enumerate(jac.values):
            for i in bins.searchsorted(abs(j)):
                hist[t, i-1] += 1
        plt.colorbar(plt.imshow(hist[500:800, :-1].T, cmap='coolwarm'))
        plt.show()


if __name__ == '__main__':
    climate.call(main)
