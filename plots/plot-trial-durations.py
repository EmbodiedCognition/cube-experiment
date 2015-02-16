#!/usr/bin/env python

import climate
import collections
import lmj.cubes
import lmj.plot
import numpy as np


@lmj.cubes.utils.pickled
def durations(root, pattern):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in lmj.cubes.Experiment(root).subjects:
        for i, b in enumerate(s.blocks):
            for j, t in enumerate(b.trials):
                if t.matches(pattern):
                    t.load()
                    data[i][j].append(t.df.index[-1])
    for i in data:
        data[i] = dict(data[i])
    return dict(data)


def set_y_stuff(ax, yticks=(30, 40, 50, 60, 70)):
    ax.set_ylabel('Trial Duration (sec)')
    ax.set_ylim(28, 72)
    ax.set_yticks(yticks)
    for y in yticks:
        ax.axhline(y, -1, 40, linestyle=':', color='#111111', alpha=0.3, lw=1)


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'))
def main(root, pattern='*'):
    data = durations(root, pattern)

    with lmj.plot.axes(spines=True) as ax:
        t = 0
        labels = []
        ticks = []
        for i in sorted(data):
            means = []
            stderrs = []
            for j in sorted(data[i]):
                if j == 0:
                    t += 1
                ticks.append(t + j)
                labels.append(str(j + 1))
                d = np.array(data[i][j])
                # discard outliers more than 3 stdevs from mean
                e = d[abs((d - d.mean()) / d.std()) < 3]
                means.append(e.mean())
                stderrs.append(e.std() / np.sqrt(len(e)))
                ax.scatter(t + j + 0.05 * np.random.randn(len(d)), d,
                           c='#111111', facecolors='none', lw=1, alpha=0.7)
            ax.errorbar(range(t, t + len(means)), means, yerr=stderrs, color='#111111', lw=2)
            t += len(means)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Trial')
        ax.set_xlim(0.5, len(ticks) - 0.5)
        set_y_stuff(ax)

    with lmj.plot.axes(spines=True) as ax:
        means = []
        stderrs = []
        for i in sorted(data):
            if i == 5:
                break
            d = np.array(data[i][0])
            # discard outliers more than 3 stdevs from mean
            e = d[abs((d - d.mean()) / d.std()) < 3]
            means.append(e.mean())
            stderrs.append(e.std() / np.sqrt(len(e)))
            ax.scatter(i + 0.05 * np.random.randn(len(d)), d,
                       c='#111111', facecolors='none', lw=1, alpha=0.9)
        ax.errorbar(range(len(means)), means, yerr=stderrs, color='#111111', lw=2)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
        ax.set_xlabel('Block')
        ax.set_xlim(-0.5, 4.5)
        set_y_stuff(ax)


if __name__ == '__main__':
    climate.call(main)
