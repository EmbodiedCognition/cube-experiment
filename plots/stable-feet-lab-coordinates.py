import climate
import collections
import lmj.cubes
import lmj.plot
import numpy as np
import pandas as pd

import utils


def main(root, pattern='*'):
    points = collections.defaultdict(list)
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        trial.add_velocities(7)
        for m, n in (('r', 40), ('l', 47)):
            t = trial.trajectory(n, velocity=True)
            t['speed'] = np.sqrt((t[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
            for i in utils.local_minima(df.speed.values):
                points[m].append(t.iloc[i, :])

    for m in sorted(points):
        print(m, len(points[m]))

    left_label = True
    right_label = True
    kw = dict(s=30, vmin=0, lw=0, alpha=0.5)
    with lmj.plot.axes(aspect='equal') as ax:
        ax.scatter([0], [0], color='#111111', alpha=0.5, s=200, marker='s', lw=0)
        for m, dfs in points.items():
            kw['cmap'] = utils.OCM if m == 'l' else utils.BCM
            for df in dfs:
                label = None
                if m == 'l':
                    if left_label:
                        label = 'Left Foot'
                        left_label = False
                if m == 'r':
                    if right_label:
                        label = 'Right Foot'
                        right_label = False
                ax.scatter(df.x, df.z, c=df.speed, label=label, **kw)
        lmj.plot.legend(loc='upper left')


if __name__ == '__main__':
    climate.call(main)
