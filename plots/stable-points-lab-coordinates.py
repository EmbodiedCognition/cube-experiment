import climate
import collections
import lmj.cubes
import lmj.plot
import matplotlib.colors
import numpy as np


KCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.0, 0.0), (1, 0.0, 0.0)),
    green=((0, 0.0, 0.0), (1, 0.0, 0.0)),
    blue= ((0, 0.0, 0.0), (1, 0.0, 0.0)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


RCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.8, 0.8), (1, 0.8, 0.8)),
    green=((0, 0.1, 0.1), (1, 0.1, 0.1)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 0.7, 0.7), (1, 0.0, 0.0)),
))


GCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.6, 0.6), (1, 0.6, 0.6)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 0.7, 0.7), (1, 0.0, 0.0)),
))


BCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 0.7, 0.7), (1, 0.0, 0.0)),
))


OCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.0, 0.0), (1, 0.0, 0.0)),
    alpha=((0, 0.7, 0.7), (1, 0.0, 0.0)),
))


PCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.6, 0.6), (1, 0.6, 0.6)),
    green=((0, 0.4, 0.4), (1, 0.4, 0.4)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 0.7, 0.7), (1, 0.0, 0.0)),
))


CMAPS = {
    'marker00-r-head-back': KCM,
    'marker01-r-head-front': KCM,
    'marker02-l-head-front': KCM,
    'marker03-l-head-back': KCM,

    'marker31-sternum': KCM,

    'marker07-r-shoulder': PCM,
    'marker13-r-fing-index': PCM,
    'marker14-r-mc-outer': PCM,
    'marker19-l-shoulder': RCM,
    'marker25-l-fing-index': RCM,
    'marker26-l-mc-outer': RCM,

    'marker34-l-ilium': GCM,
    'marker35-r-ilium': GCM,
    'marker36-r-hip': GCM,
    'marker43-l-hip': GCM,

    'marker40-r-heel': BCM,
    'marker41-r-mt-outer': BCM,
    'marker42-r-mt-inner': BCM,

    'marker47-l-heel': OCM,
    'marker48-l-mt-outer': OCM,
    'marker49-l-mt-inner': OCM,
}


WHITELIST = ('40', '41', '42', '47', '48', '49', )  # '13')


# fewf, http://stackoverflow.com/questions/4494404
def contig(cond):
    idx = np.diff(cond).nonzero()[0] + 1
    if cond[0]:
        idx = np.r_[0, idx]
    if cond[-1]:
        idx = np.r_[idx, cond.size]
    return idx.reshape((-1, 2))


def main(root, pattern='*', threshold=100):
    points = collections.defaultdict(list)
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        trial.add_velocities(7)
        for m in trial.marker_columns:
            if m[6:8] not in WHITELIST:
                continue
            t = trial.trajectory(m, velocity=True)
            t['vel'] = np.sqrt((t[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
            for s, e in contig((t.vel < threshold).values):
                if 2 < e - s < 1000:
                    points[m].append(t.iloc[s:e, :])

    for m in sorted(points):
        print(m, len(points[m]))

    kw = dict(s=30, vmin=0, vmax=1.2, lw=0, alpha=None)
    with lmj.plot.axes(aspect='equal') as ax:
        ax.scatter([0], [0], color='#111111', alpha=0.5, s=200, marker='s', lw=0)
        for m, dfs in points.items():
            for df in dfs:
                ax.scatter(df.x, df.z, c=np.log(1 + df.vel), cmap=CMAPS[m], **kw)
        #ax.set_xlim(-1, 5)
        #ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
        #ax.set_ylim(-2, 1)
        #ax.set_yticks([-2, -1, 0, 1])


if __name__ == '__main__':
    climate.call(main)
