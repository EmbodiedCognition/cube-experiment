#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np

MARKERS = 'r-fing-index l-fing-index r-heel r-head-front r-hip'

@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
    output=('save movie in this output filename', 'option'),
)
def main(root, pattern='block03*/*trial00', dropouts=None, output=None):
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
    anim = lmj.plot.rotate_3d(
        lmj.cubes.plots.show_3d(render),
        output=output, azim=(0, 90), elev=(5, 20), fig=dict(figsize=(10, 4.8)))
    if not output:
        lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
