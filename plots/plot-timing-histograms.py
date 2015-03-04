#!/usr/bin/env python

import climate
import joblib
import lmj.cubes
import lmj.plot
import numpy as np


def diffs(t):
    t.load()
    return 1000 * np.diff(t.index.values)


def main(root, pattern='*'):
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    values = joblib.Parallel(-1)(joblib.delayed(diffs)(t) for t in trials)
    values = [x for xs in values for x in xs]
    with lmj.plot.axes(spines=True, max_xticks=6, max_yticks=6) as ax:
        ax.hist(values, bins=np.linspace(5, 15, 127), lw=0, stacked=True)
        ax.set_xlim(5, 15)
        ax.set_xlabel('Time Between Frames (msec)')
        ax.set_ylabel('Number of Observations')


if __name__ == '__main__':
    climate.call(main)
