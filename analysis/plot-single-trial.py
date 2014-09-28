import climate
import lmj.plot

import source
import plots


@climate.annotate(
    root='load experiment data from this directory',
    pattern='plot data from files matching this pattern',
)
def main(root, pattern='*block00/*circuit00.csv.gz'):
    with plots.space() as ax:
        for t in source.Experiment(root).trials:
            if t.matches(pattern):
                t.load()
                x, y, z = t.marker('r-fing-index')
                ax.plot(x, z, zs=y, color=lmj.plot.COLOR11[0], alpha=0.7)
                x, y, z = t.marker('l-fing-index')
                ax.plot(x, z, zs=y, color=lmj.plot.COLOR11[1], alpha=0.7)
                x, y, z = t.marker('r-heel')
                ax.plot(x, z, zs=y, color=lmj.plot.COLOR11[2], alpha=0.7)
                x, y, z = t.marker('r-knee')
                ax.plot(x, z, zs=y, color=lmj.plot.COLOR11[3], alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
