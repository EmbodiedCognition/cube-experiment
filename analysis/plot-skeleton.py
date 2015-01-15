#!/usr/bin/env python

import climate

import database
import plots


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block02/*trial00*.csv.gz'):
    for trial in database.Experiment(root).trials_matching(pattern):
        with plots.space() as ax:
            plots.skeleton(ax, trial, 100)


if __name__ == '__main__':
    climate.call(main)
