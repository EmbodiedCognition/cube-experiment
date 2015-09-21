import climate
import collections
import lmj.cubes
import lmj.plot
import matplotlib.colors
import numpy as np
import pandas as pd


COLORS = {
    'marker00-r-head-back': '#9467bd',
    'marker01-r-head-front': '#9467bd',
    'marker02-l-head-front': '#9467bd',
    'marker03-l-head-back': '#9467bd',
    'marker07-r-shoulder': '#111111',
    'marker13-r-fing-index': '#111111',
    'marker14-r-mc-outer': '#111111',
    'marker19-l-shoulder': '#111111',
    'marker25-l-fing-index': '#111111',
    'marker26-l-mc-outer': '#111111',
    'marker31-sternum': '#111111',
    'marker34-l-ilium': '#2ca02c',
    'marker35-r-ilium': '#2ca02c',
    'marker36-r-hip': '#2ca02c',
    'marker40-r-heel': '#1f77b4',
    'marker41-r-mt-outer': '#1f77b4',
    'marker42-r-mt-inner': '#1f77b4',
    'marker43-l-hip': '#2ca02c',
    'marker47-l-heel': '#d62728',
    'marker48-l-mt-outer': '#d62728',
    'marker49-l-mt-inner': '#d62728',
}


RCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.8, 0.8), (1, 0.8, 0.8)),
    green=((0, 0.1, 0.1), (1, 0.1, 0.1)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


GCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.6, 0.6), (1, 0.6, 0.6)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


BCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


OCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 1.0, 1.0), (1, 1.0, 1.0)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.0, 0.0), (1, 0.0, 0.0)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


PCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.6, 0.6), (1, 0.6, 0.6)),
    green=((0, 0.4, 0.4), (1, 0.4, 0.4)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


# fewf, http://stackoverflow.com/questions/4494404
def contig(cond):
    idx = np.diff(cond).nonzero()[0] + 1
    if cond[0]:
        idx = np.r_[0, idx]
    if cond[-1]:
        idx = np.r_[idx, cond.size]
    return idx.reshape((-1, 2))


def main(root, pattern='*'):
    points = collections.defaultdict(list)
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        trial.add_velocities(7)

        lheel = trial.trajectory(40, velocity=True)
        lheel['speed'] = np.sqrt((lheel[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
        lslow = contig((lheel.speed < 1).values)

        rheel = trial.trajectory(47, velocity=True)
        rheel['speed'] = np.sqrt((rheel[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
        #rslow = contig((rheel.speed < 2).values)

        for m in trial.marker_columns:
            t = trial.trajectory(m, velocity=True)
            t['speed'] = np.sqrt((t[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
            for s, e in lslow:
                if e - s > 0:
                    points[m].append(t.iloc[s:e, :])

    for m in sorted(points):
        print(m, len(points[m]))

    kw = dict(s=30, vmin=0, vmax=1, lw=0, alpha=0.5, cmap=BCM)
    with lmj.plot.axes3d() as ax:
        ax.scatter([0], [0], [0], color='#111111', alpha=0.5, s=200, marker='s', lw=0)
        for m, dfs in points.items():
            for df in dfs:
                # subsample.
                sel = np.random.rand(len(df)) < 0.01
                ax.scatter(df[sel].x, df[sel].z, df[sel].y, c=df[sel].speed, **kw)


if __name__ == '__main__':
    climate.call(main)
