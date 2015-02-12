#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np

MARKERS = 'r-fing-index l-fing-index r-heel r-knee'

@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot traces of these markers', 'option'),
    dropouts=('replace dropout frames with nans', 'option'),
)
def main(root, pattern='*/*block03*/*trial00*.csv.gz', markers=MARKERS, dropouts=None):
    cubes = True
    with lmj.cubes.plots.space() as ax:
        for t in lmj.cubes.Experiment(root).trials_matching(pattern):
            t.load()
            if cubes:
                lmj.cubes.plots.show_cubes(ax, t)
                cubes = False
            if dropouts:
                t.mask_dropouts()
            for i, marker in enumerate(markers.split()):
                df = t.trajectory(marker)
                ax.plot(np.asarray(df.x),
                        np.asarray(df.z),
                        zs=np.asarray(df.y),
                        color=lmj.plot.COLOR11[i],
                        alpha=0.7,
                        lw=2,
                )


if __name__ == '__main__':
    climate.call(main)
