#!/usr/bin/env python

import climate
import lmj.cubes

@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
    targets=('only plot these targets', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz', targets=None, output=None):
    targets = set(range(12))
    if targets is not None:
        targets = set(int(x.strip()) for x in targets.strip().split(','))
    with lmj.cubes.plots.space() as ax:
        cubes = True
        for t in lmj.cubes.Experiment(root).trials_matching(pattern):
            t.load()
            t.add_velocities(smooth=7)
            if cubes:
                lmj.cubes.plots.show_cubes(ax, t)
                cubes = False
            for T in targets:
                mov = t.movement_to(T)
                if len(mov.df):
                    lmj.cubes.plots.skeleton(ax, mov, -1, lw=2)


if __name__ == '__main__':
    climate.call(main)
