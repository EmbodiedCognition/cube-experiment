#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np

MARKERS = 'r-fing-index l-fing-index r-heel r-head-front'

@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot traces of these markers', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(root, pattern='*/*block03*/*trial00*.csv.gz', markers=MARKERS, dropouts=None, output=None):
    cubes = True
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
                for i, marker in enumerate(markers.split()):
                    df = t.trajectory(marker)
                    ax.plot(np.asarray(df.x)[4:-4],
                            np.asarray(df.z)[4:-4],
                            zs=np.asarray(df.y)[4:-4],
                            color=lmj.plot.COLOR11[i],
                            alpha=0.7,
                            lw=2,
                    )
        return render
    anim = lmj.plot.rotate_3d(
        show, output=output,
        azim=(0, 90), elev=(5, 20),
        fig=dict(figsize=(10, 4.8)),
        movie_args=('-vcodec', 'libx264', '-b', '3000k'),
    )
    if not output:
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
