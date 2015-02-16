#!/usr/bin/env python

import climate
import lmj.plot
import numpy as np


def main(data='target-movement-durations.csv'):
    data = np.loadtxt(data, delimiter=',')
    with lmj.plot.axes(spines=True) as ax:
        xs = []
        for i, c in enumerate(data.T):
            x = c[c > 10] / 100
            z = (x - x.mean()) / x.std()
            xs.append(x)
            print('{:.0f} {:.1f} {:.0f} {:.1f} -- {} {}'.format(
                x.min(), x.mean(), x.max(), x.std(),
                sum(abs(z) > 2), sum(abs(z) > 3)))
            ax.scatter(i + 0.1 * np.random.randn(len(x)), x,
                       facecolor='#111111', alpha=0.1, lw=0.01)
        kw = dict(color='#d62728', linestyle='-', lw=2, alpha=0.9)
        ax.boxplot(xs,
                   sym='',
                   positions=np.arange(12),
                   boxprops=kw,
                   whiskerprops=kw,
                   capprops=kw,
                   medianprops=kw,
                   widths=0.6,
                   whis=[10, 90])
        ax.set_xlim(-0.5, 11.5)
        ax.set_xticks(range(12))
        ax.set_xticklabels(range(12))
        ax.set_ylim(0, 10)
        ax.set_ylabel('Segment Duration (sec)')
        ax.set_xlabel('Target')


if __name__ == '__main__':
    climate.call(main)
