#!/usr/bin/env python

import climate
import collections
import joblib
import lmj.cubes
import lmj.plot
import numpy as np

logging = climate.get_logger('count')


def count(trial):
    trial.load()
    trial.mask_dropouts()
    total = len(trial.df)
    markers = {m: trial.df[m + '-c'].count() / total for m in trial.marker_columns}
    full = len(trial.df[[m + '-c' for m in markers]].dropna(axis=0))
    trial.log('%d rows, %d full (%.1f%%)', total, full, 100 * full / total)
    return markers


PERCENTILES = [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99]


def main(root):
    trials = lmj.cubes.Experiment(root).trials_matching('*')
    counts = collections.defaultdict(int)
    percents = collections.defaultdict(list)
    f = joblib.delayed(count)
    for markers in joblib.Parallel(-1)(f(t) for t in trials):
        for m in markers:
            counts[m] += markers[m] > 0.1
            percents[m].append(markers[m])
    print(*(['marker', 'count'] + [str(x) for x in PERCENTILES]), sep='\t')
    for m, c in counts.items():
        print(m, c, *np.percentile(percents[m], PERCENTILES), sep='\t')
    return
    with lmj.plot.axes(spines=True) as ax:
        for m, values in percents.items():
            ax.hist(values, bins=np.linspace(0, 1, 127), alpha=0.5, lw=0, label=m[9:])
        ax.legend(ncol=3, loc=0)


if __name__ == '__main__':
    climate.call(main)
