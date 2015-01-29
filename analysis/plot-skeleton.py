#!/usr/bin/env python

import climate
import pandas as pd

import database
import plots


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block03/*trial00*.csv.gz'):
    for trial in database.Experiment(root).trials_matching(pattern):
        with plots.space() as ax:
            for i in range(3):
                plots.skeleton(ax, trial, 1000 + 300 * i, lw=2, color='#fd3220', alpha=0.3)
            #trial.rotate_heading(pd.Series([-6.28 / 10] * len(trial.df)))
            trial.make_body_relative()
            for i in range(3):
                plots.skeleton(ax, trial, 1000 + 300 * i, offset=(0.5 * i, 0.5 * i), lw=2, color='#111111', alpha=0.3)


if __name__ == '__main__':
    climate.call(main)
