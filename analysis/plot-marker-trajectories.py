import climate
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot traces of these markers', 'option'),
    spline=('interpolate data with a spline of this order', 'option', None, int),
    accuracy=('fit spline with this accuracy', 'option', None, float),
    svt_threshold=('trajectory-SVT threshold', 'option', None, float),
    svt_frames=('number of trajectory-SVT frames', 'option', None, int),
)
def main(root,
         pattern='*/*block00/*circuit00.csv.gz',
         markers='r-fing-index l-fing-index r-heel r-knee',
         spline=None,
         accuracy=0.01,
         svt_threshold=1000,
         svt_frames=5):
    with plots.space() as ax:
        for t in database.Experiment(root).trials_matching(pattern):
            if spline:
                t.normalize(order=spline, accuracy=accuracy)
            else:
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
