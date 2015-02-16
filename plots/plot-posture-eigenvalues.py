#!/usr/bin/env python

import climate
import lmj.pca
import lmj.plot


def main(root, pca):
    pca = lmj.pca.PCA(filename=pca)

    K = 100
    annkw = dict(
        color='#d62728',
        arrowprops=dict(
            color='#d62728',
            arrowstyle='->',
            connectionstyle='angle,angleA=-100,angleB=10,rad=10'))
    with lmj.plot.axes(spines=True) as ax:
        ax.bar(range(K), (pca.vals / pca.vals.sum())[:K], color='#111111', lw=0)
        ax.set_xlim(0, K)
        ax.set_yticks([0.1, 0.2, 0.3])
        ax.grid(True)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Fraction of Variance')
        for c in (0.5, 0.9, 0.99):
            k = pca.num_components(c)
            s = '{:.0f}%'.format(100 * c)
            ax.axvline(k, 0, pca.vals[0], color='#d62728', lw=1)
            ax.annotate(s, xy=(k, 0.26), xytext=(k + 2, 0.28), **annkw)


if __name__ == '__main__':
    climate.call(main)
