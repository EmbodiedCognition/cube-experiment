#!/usr/bin/env python

import climate
import lmj.plot
import numpy as np


def main(data='target-movement-durations.csv'):
    data = np.loadtxt(data, delimiter=',')
    with lmj.plot.with_axes(spines=True) as ax:
        xs = []
        for i, c in enumerate(data.T):
            x = c[c > 10] / 100
            z = (x - x.mean()) / x.std()
            x = x[abs(z) < 3]
            xs.append(x)
            print('{:.0f} {:.1f} {:.0f} {:.1f} -- {} {}'.format(
                x.min(), x.mean(), x.max(), x.std(),
                sum(abs(z) > 2), sum(abs(z) > 3)))
            ax.scatter(1 + i + 0.1 * np.random.randn(len(x)),
                       x,
                       facecolor=lmj.plot.COLOR11[i % 11],
                       alpha=0.5,
                       lw=0.01)
        kw = dict(color='#666666', linestyle='-', lw=2)
        ax.boxplot(xs, sym='',
                   boxprops=kw,
                   whiskerprops=kw,
                   capprops=kw,
                   medianprops=kw,
                   widths=0.8,
                   whis=[2, 98])
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0, 10)
        ax.set_ylabel('Movement Duration (sec)')
        ax.set_xlabel('Target')


if __name__ == '__main__':
    climate.call(main)
