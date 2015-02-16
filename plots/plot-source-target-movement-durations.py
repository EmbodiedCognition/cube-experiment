#!/usr/bin/env python

import climate
import lmj.plot
import numpy as np


def main(data='source-target-movement-durations.csv'):
    data = np.loadtxt(data, delimiter=',')
    with lmj.plot.axes(spines=True) as ax:
        xs = []
        col = []
        pos = []
        sources = set()
        for i, c in enumerate(data.T):
            s, t = divmod(i, 12)
            x = c[c > 10] / 100
            if len(x) == 0:
                continue
            z = (x - x.mean()) / x.std()
            xs.append(x[abs(z) < 3])
            col.append(s)
            pos.append(t)
            print('{:.0f} {:.1f} {:.0f} {:.1f} -- {} {}'.format(
                x.min(), x.mean(), x.max(), x.std(),
                sum(abs(z) > 2), sum(abs(z) > 3)))
            kw = dict()
            if s not in sources:
                kw['label'] = 'Source {}'.format(1 + s)
                sources.add(s)
            ax.scatter(t + 0.1 * np.random.randn(len(x)), x,
                       facecolor='#111111', alpha=0.1, lw=0.01, **kw)
        for s in range(12):
            kw = dict(color=lmj.plot.COLOR11[s % 11], linestyle='-', lw=2, alpha=0.9)
            ax.boxplot(
                [x for i, x in enumerate(xs) if col[i] == s],
                sym='',
                positions=[p + 0.1 * np.random.randn() for i, p in enumerate(pos) if col[i] == s],
                boxprops=kw,
                whiskerprops=kw,
                capprops=kw,
                medianprops=kw,
                widths=0.6,
                whis=[10, 90])
        #ax.legend()
        ax.set_xlim(-0.5, 11.5)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels([str(x) for x in np.arange(12)])
        ax.set_xlabel('Target')
        ax.set_ylim(0, 10)
        ax.set_ylabel('Segment Duration (sec)')


if __name__ == '__main__':
    climate.call(main)
