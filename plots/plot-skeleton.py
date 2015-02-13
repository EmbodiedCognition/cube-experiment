#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz', targets=(10, )):
    cubes = True
    with lmj.cubes.plots.space() as ax:
        for i, trial in enumerate(lmj.cubes.Experiment(root).trials_matching(pattern)):
            trial.load()
            if cubes:
                lmj.cubes.plots.show_cubes(ax, trial, targets)
                cubes = False
            for t in targets:
                mov = trial.movement_to(t)
                if len(mov.df):
                    lmj.cubes.plots.skeleton(ax, mov, -1, lw=2)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')


if __name__ == '__main__':
    climate.call(main)
