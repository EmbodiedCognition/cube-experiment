#!/usr/bin/env python

import climate
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz'):
    cubes = True
    with plots.space() as ax:
        for i, trial in enumerate(database.Experiment(root).trials_matching(pattern)):
            trial.load()
            #trial.mask_fiddly_target_frames()
            #trial.mask_dropouts()
            if cubes:
                plots.show_cubes(ax, trial)
                cubes = False
            df = trial.trajectory('r-fing-index')
            ax.plot(np.asarray(df.x),
                    np.asarray(df.z),
                    zs=np.asarray(df.y),
                    color=lmj.plot.COLOR11[1],
                    alpha=0.7,
                    lw=2)
            plots.skeleton(ax, trial.movement_to(6), -1, color='#111111', lw=2)
            ax.azim = -60
            ax.elev = 11
            ax.dist = 8


if __name__ == '__main__':
    climate.call(main)
