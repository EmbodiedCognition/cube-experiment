import climate
import collections
import lmj.cubes
import lmj.plot
import numpy as np


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
        trial.add_velocities(3)
        for m in trial.marker_columns:
            t = trial.trajectory(m, velocity=True)
            vel = np.sqrt((t[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
            for s, e in contig((vel < 0.01).values):
                if e - s > 3:
                    points[m].append(t.iloc[(s + e) // 2, :])
    for m in sorted(points):
        print(m, len(points[m]))
    with lmj.plot.axes() as ax:
        for m, dfs in points.items():
            if len(dfs) > 2:
                for df in dfs:
                    ax.scatter(df.x, df.z, color=COLORS[m], s=20, alpha=0.9, lw=0)


if __name__ == '__main__':
    climate.call(main)
