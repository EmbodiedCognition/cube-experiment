#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot

TARGETS = (0, 1, 4)

@climate.annotate(
    root='plot data rooted at this path',
    pattern=('plot data from files matching this pattern', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(root, pattern='*/*block01/*trial01*.csv.gz', dropouts=None, output=None):
    def show(ax):
        ax.w_xaxis.set_pane_color((1, 1, 1, 1))
        ax.w_yaxis.set_pane_color((1, 1, 1, 1))
        ax.w_zaxis.set_pane_color((1, 1, 1, 1))
        ax.set_xlim(-2, 2)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels([])
        ax.set_ylim(-2, 2)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels([])
        ax.set_zlim(0, 2)
        ax.set_zticks([0, 1, 2])
        ax.set_zticklabels([])
        def render():
            cubes = True
            for t in lmj.cubes.Experiment(root).trials_matching(pattern):
                t.load()
                if cubes:
                    lmj.cubes.plots.show_cubes(ax, t)
                    cubes = False
                if dropouts:
                    t.mask_dropouts()
                for n in TARGETS:
                    mov = t.movement_to(n)
                    if len(mov.df):
                        lmj.cubes.plots.skeleton(ax, mov, -1, lw=2)
        return render
    anim = lmj.plot.rotate_3d(
        show, output=output,
        azim=(0, 90), elev=(5, 20),
        fig=dict(figsize=(10, 4.8)),
    )
    if not output:
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
