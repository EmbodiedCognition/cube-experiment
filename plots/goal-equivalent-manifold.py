#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot


@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
    targets=('only plot these targets', 'option'),
    output=('save result in this output filename', 'option'),
    animate=('create animated output', 'option'),
)
def main(root, pattern='*', targets=None, output=None, animate=None):
    ts = set(range(12))
    if targets is not None:
        ts = set(int(x.strip()) for x in targets.strip().split(','))
    targets = ts

    def render(ax):
        cubes = True
        for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
            trial.load()
            #trial.add_velocities(smooth=7)
            if cubes:
                lmj.cubes.plots.show_cubes(ax, trial)
                cubes = False
            for target in targets:
                mov = trial.movement_to(target)
                if len(mov.df):
                    lmj.cubes.plots.skeleton(ax, mov, -1, vel=False, lw=2, alpha=0.1)

    if animate:
        lmj.plot.rotate_3d(
            lmj.cubes.plots.show_3d(render),
            output=output,
            azim=(0, 100),
            elev=(10, 20),
            fig=dict(figsize=(10, 4.8)))
        if not output:
            lmj.plot.show()
    else:
        with lmj.cubes.plots.space() as ax:
            render(ax)


if __name__ == '__main__':
    climate.call(main)
