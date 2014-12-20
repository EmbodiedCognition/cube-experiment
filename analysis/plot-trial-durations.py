#!/usr/bin/env python

import climate
import collections
import lmj.plot
import numpy as np

import database
import plots


@database.pickled
def durations(root, pattern):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in database.Experiment(root).subjects:
        for i, b in enumerate(s.blocks):
            for j, t in enumerate(b.trials):
                if t.matches(pattern):
                    t.load()
                    data[i][j].append(t.df.index[-1] / t.total_distance)
    for i in data:
        data[i] = dict(data[i])
    return dict(data)


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'))
def main(root, pattern='*.csv.gz'):
    data = durations(root, pattern)
    labels = []
    with lmj.plot.with_axes(spines=True) as ax:
        t = 0
        for i in sorted(data):
            means = []
            stds = []
            for j in sorted(data[i]):
                labels.append(str(j + 1))
                d = np.array(data[i][j])
                z = (d - d.mean()) / d.std()
                # discard outliers more than 3 stdevs from mean
                valid = d[abs(z) < 3]
                means.append(valid.mean())
                stds.append(valid.std())
                ax.scatter(len(labels) - 1 + 0.1 * np.random.randn(len(valid)), valid,
                           c=lmj.plot.COLOR11[0], facecolors='none', lw=1, alpha=0.7)
            u = t + len(means)
            ax.errorbar(range(t, u), means, yerr=stds, color=lmj.plot.COLOR11[0], lw=2)
            t = u
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Trial')
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel('Normalized Duration (s/m)')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['1', '2', '3'])


if __name__ == '__main__':
    climate.call(main)
