#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np

MARKERS = 'r-fing-index l-fing-index r-heel r-head-front'

@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
    output=('save result in this output filename', 'option'),
    animate=('make an animated, rotating graph', 'option'),
)
def main(root, pattern='block03*/*trial00', dropouts=None, output=None, animate=None):
    def render(ax):
        cubes = True
        for t in lmj.cubes.Experiment(root).trials_matching(pattern):
            t.load()
            if cubes:
                lmj.cubes.plots.show_cubes(ax, t)
                cubes = False
            if dropouts:
                t.mask_dropouts()
            for i, marker in enumerate(MARKERS.split()):
                df = t.trajectory(marker)
                ax.plot(np.asarray(df.x),
                        np.asarray(df.z),
                        zs=np.asarray(df.y),
                        color=lmj.plot.COLOR11[i],
                        alpha=0.7, lw=2)
    if animate:
        lmj.plot.rotate_3d(
            lmj.cubes.plots.show_3d(render),
            output=output,
            azim=(0, 90),
            elev=(5, 20),
            fig=dict(figsize=(10, 4.8)))
        if not output:
            lmj.plot.show()
    else:
        with lmj.plot.axes3d() as ax:
            render(ax)
            lmj.cubes.plots.show_3d(lambda ax: None)(ax)
            ax.view_init(elev=15, azim=-110)


if __name__ == '__main__':
    climate.call(main)
