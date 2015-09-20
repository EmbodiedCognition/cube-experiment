#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.pca
import lmj.plot
import numpy as np
import pandas as pd


@climate.annotate(
    pca='load PCA from this file on disk',
    root='load experiment data from this directory tree',
    pattern=('load trials matching this pattern', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(pca, root, pattern='*', output=None):
    trial = next(lmj.cubes.Experiment(root).trials_matching(pattern))
    trial.load()
    trial.add_velocities()

    pca = lmj.pca.PCA(filename=pca)

    '''
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
    '''

    mov = lmj.cubes.Movement(pd.DataFrame(
        (pca.vals * pca.vecs).T, columns=trial.marker_channel_columns))
    def render(ax):
        lmj.cubes.plots.skeleton(ax, mov, 10, lw=2)
    anim = lmj.plot.rotate_3d(
        lmj.cubes.plots.show_3d(render),
        output=output, azim=(0, 90), elev=(5, 20), fig=dict(figsize=(10, 4.8)))
    if not output:
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
