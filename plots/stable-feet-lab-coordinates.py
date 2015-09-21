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
            points[m].append(t.iloc[utils.local_minima(t.speed.values), :])

    for m in sorted(points):
        print(m, len(points[m]))

    kw = dict(markersize=10, marker='o', linestyle='-', lw=1, alpha=0.5)
    with lmj.plot.axes(aspect='equal') as ax:
        ax.scatter([0], [0], color='#111111', alpha=0.7, s=200, marker='s', lw=0)
        for m, dfs in points.items():
            kw['color'] = '#ff9900' if m == 'l' else '#3232c0'
            kw['label'] = 'Left Foot' if m == 'l' else 'Right Foot'
            for df in dfs:
                ax.plot(df.x, df.z, **kw)
                kw['label'] = None
        lmj.plot.legend(loc='upper left')


if __name__ == '__main__':
    climate.call(main)
