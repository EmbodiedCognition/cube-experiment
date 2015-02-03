#!/usr/bin/env python

import climate

import database
import plots


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block03/*trial00*.csv.gz', targets=(0, 1, 2, 3)):
    for trial in database.Experiment(root).trials_matching(pattern):
        trial.load()
        with plots.space() as ax:
            plots.show_cubes(ax, trial, targets)
            for t in targets:
                mov = trial.movement_to(t)
                if len(mov.df):
                    for i in range(1, 5):
                        plots.skeleton(ax, mov, -10 * i, lw=2, color='#111111', alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
