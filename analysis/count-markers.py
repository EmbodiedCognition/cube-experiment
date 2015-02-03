#!/usr/bin/env python

import climate
import collections
import joblib

import database


def count(trial):
    trial.load()
    trial.reindex(100)
    trial.mask_dropouts()
    markers = []
    for m in trial.marker_columns:
        s = trial[m + '-c']
        if s.count() > 0.01 * len(s):
            markers.append(m)
    return markers


def main(root):
    trials = database.Experiment(root).trials_matching('*')
    counts = collections.defaultdict(int)
    f = joblib.delayed(count)
    for markers in joblib.Parallel(-1)(f(t) for t in trials):
        for m in markers:
            counts[m] += 1
    for m, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(c, m)


if __name__ == '__main__':
    climate.call(main)
