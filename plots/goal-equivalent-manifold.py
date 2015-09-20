#!/usr/bin/env python

import climate
import lmj.cubes

@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
    targets=('only plot these targets', 'option'),
)
def main(root, pattern='*', targets=None):
    ts = set(range(12))
    if targets is not None:
        ts = set(int(x.strip()) for x in targets.strip().split(','))
    targets = ts
    with lmj.cubes.plots.space() as ax:
        cubes = True
        for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
            trial.load()
            trial.add_velocities(smooth=7)
            if cubes:
                lmj.cubes.plots.show_cubes(ax, trial)
                cubes = False
            for target in targets:
                mov = trial.movement_to(target)
                if len(mov.df):
                    lmj.cubes.plots.skeleton(ax, mov, -1, lw=2, alpha=0.1)


if __name__ == '__main__':
    climate.call(main)
