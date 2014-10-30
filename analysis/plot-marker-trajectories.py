import climate
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot traces of these markers', 'option'),
    accuracy=('fit spline/SVT with this accuracy', 'option', None, float),
    spline_order=('interpolate data with a spline of this order', 'option', None, int),
    svt_threshold=('trajectory-SVT threshold', 'option', None, float),
    svt_frames=('number of trajectory-SVT frames', 'option', None, int),
)
def main(root,
         pattern='*/*block03*/*trial00*.csv.gz',
         markers='r-fing-index l-fing-index r-heel r-knee',
         cubes='yes',
         accuracy=0.002,
         spline_order=None,
         svt_threshold=None,
         svt_frames=5):
    with plots.space() as ax:
        for t in database.Experiment(root).trials_matching(pattern):
            if cubes:
                plots.show_cubes(ax, t)
                cubes = False
            if spline_order:
                t.normalize(order=spline_order, accuracy=1. / accuracy)
            elif svt_threshold:
                t.reindex()
                t.svt(svt_threshold, accuracy, svt_frames)
            for i, marker in enumerate(markers.split()):
                df = t.trajectory(marker)
                ax.plot(np.asarray(df.x),
                        np.asarray(df.z),
                        zs=np.asarray(df.y),
                        color=lmj.plot.COLOR11[i],
                        alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
