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
        #for m in trial.marker_columns:
        #    if m[:8] not in ('marker40', 'marker41', 'marker42',
        #                     'marker47', 'marker48', 'marker49'):
        #        continue
        for m, h, o, i in (('-r-', 40, 41, 42), ('-l-', 47, 48, 49)):
            th = trial.trajectory(h, velocity=True)
            to = trial.trajectory(o, velocity=True)
            ti = trial.trajectory(i, velocity=True)
            t = pd.DataFrame(index=th.index)
            t['vel'] = np.sqrt((th[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
            t['ox'] = to.x
            t['oz'] = to.z
            t['ix_'] = ti.x
            t['iz'] = ti.z
            t['hx'] = th.x
            t['hz'] = th.z
            for s, e in contig((t.vel < 2).values):
                if e - s > 0:
                    points[m].append(t.iloc[s:e, :])

    for m in sorted(points):
        print(m, len(points[m]))

    left_label = True
    right_label = True
    kw = dict(s=30, vmin=0, vmax=2, lw=0, alpha=0.5)
    with lmj.plot.axes() as ax:
        ax.scatter([0], [0], color='#111111', alpha=0.5, s=200, marker='s', lw=0)
        for m, dfs in points.items():
            for df in dfs:
                label = col = None
                if '-l-' in m:
                    kw['cmap'] = OCM
                    col = '#ff7f0e'
                    if left_label:
                        label = 'Left Foot'
                        left_label = False
                if '-r-' in m:
                    kw['cmap'] = BCM
                    col = '#1f77b4'
                    if right_label:
                        label = 'Right Foot'
                        right_label = False
                ax.scatter(df.hx, df.hz, c=df.vel, label=label, **kw)
        lmj.plot.legend(loc='upper left')


if __name__ == '__main__':
    climate.call(main)
