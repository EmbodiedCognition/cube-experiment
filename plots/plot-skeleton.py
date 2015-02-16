#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot

TARGETS = ((0, -1), (1, -1), (8, -100), (4, -1))

@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz', dropouts=None, output=None):
    def render(ax):
        cubes = True
        for t in lmj.cubes.Experiment(root).trials_matching(pattern):
            t.load()
            t.add_velocities(smooth=3)
            if cubes:
                lmj.cubes.plots.show_cubes(ax, t)
                cubes = False
            if dropouts:
                t.mask_dropouts()
            for T, F in TARGETS:
                mov = t.movement_to(T)
                if len(mov.df):
                    lmj.cubes.plots.skeleton(ax, mov, F, lw=2)
    anim = lmj.plot.rotate_3d(
        lmj.cubes.plots.show_3d(render),
        output=output, azim=(0, 90), elev=(5, 20), fig=dict(figsize=(10, 4.8)))
    if not output:
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
