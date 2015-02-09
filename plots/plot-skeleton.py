#!/usr/bin/env python

import climate
import lmj.plot

import database
import plots


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz', targets=(0, 4, 10)):
    cubes = True
    with plots.space() as ax:
        for i, trial in enumerate(database.Experiment(root).trials_matching(pattern)):
            trial.load()
            #trial.mask_fiddly_target_frames()
            #trial.mask_dropouts()
            if cubes:
                plots.show_cubes(ax, trial, targets)
                cubes = False
            for t in targets:
                mov = trial.movement_to(t)
                if len(mov.df):
                    plots.skeleton(ax, mov, -1, lw=2, alpha=0.9)


if __name__ == '__main__':
    climate.call(main)
