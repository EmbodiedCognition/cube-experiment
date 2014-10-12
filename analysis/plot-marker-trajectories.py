import climate
import lmj.plot
import numpy as np

import source
import plots


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot traces of these markers', 'option'),
    spline=('interpolate data with a spline of this order', 'option', None, int),
    accuracy=('fit spline with this accuracy', 'option', None, float),
)
def main(root,
         pattern='*/*block00/*circuit00.csv.gz',
         markers='r-fing-index l-fing-index r-heel r-knee',
         spline=1,
         accuracy=1):
    with plots.space() as ax:
        for t in source.Experiment(root).trials_matching(pattern):
            t.normalize(order=spline, accuracy=accuracy)
            for i, marker in enumerate(markers.split()):
                df = t.trajectory(marker)
                ax.plot(np.asarray(df.x),
                        np.asarray(df.z),
                        zs=np.asarray(df.y),
                        color=lmj.plot.COLOR11[i],
                        alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
