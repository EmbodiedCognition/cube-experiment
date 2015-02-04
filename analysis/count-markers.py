#!/usr/bin/env python

import climate
import collections
import joblib

logging = climate.get_logger('count')

import database


def count(trial):
    trial.load()
    trial.mask_dropouts()
    trial.reindex(100)
    markers = []
    for m in trial.marker_columns:
        s = trial.df[m + '-c']
        if s.count() > 0.1 * len(s):
            markers.append(m)
    total = len(trial.df)
    full = len(trial.df[[m + '-c' for m in markers]].dropna(axis=0, thresh=2))
    logging.info('%s %s %s: %d rows, %d full (%.1f%%)',
                 trial.subject.key, trial.block.key, trial.key,
                 total, full, 100 * full / total)
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
