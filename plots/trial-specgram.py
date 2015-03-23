#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np


def main(root, pattern='*'):
    trials = list(lmj.cubes.Experiment(root).trials_matching(pattern))
    movs = []
    for t in trials:
        t.load()
        df = t.movement_to(9).df
        movs.append(df)
        df = df[[c for c in df.columns if c.startswith('marker') and c[-1] in 'xyz']]
        print(df.min().min(), df.mean().mean(), df.max().max())
    tmax = max(len(df) for df in movs)
    for i, (t, df) in enumerate(zip(trials, movs)):
        for j, l in enumerate('xyz'):
            loc = len(trials), 3, i * 3 + j + 1
            ax = lmj.plot.create_axes(loc, spines='bottom' if i == len(trials)-1 else False)
            cols = [c for c in t.marker_position_columns if c.endswith(l)]
            ax.imshow(df[cols].T, vmin=0 if l == 'y' else -2.5, vmax=2.5, cmap='PuOr')
            ax.set_xlim(0, tmax)
    lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
