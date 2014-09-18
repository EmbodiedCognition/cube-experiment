import climate
import lmj.plot

import source


def main(subject):
    subj = Subject(subject)
    trial = subj.blocks[0].trials[0]
    trial.load()

    ax = lmj.plot.axes(111, projection='3d')
    x, y, z = trial.marker('r-fing-index')
    ax.plot(x, z, zs=y)
    x, y, z = trial.marker('l-fing-index')
    ax.plot(x, z, zs=y)
    x, y, z = trial.marker('r-heel')
    ax.plot(x, z, zs=y)
    x, y, z = trial.marker('l-heel')
    ax.plot(x, z, zs=y)
    x, y, z = trial.marker('r-knee')
    ax.plot(x, z, zs=y)
    x, y, z = trial.marker('l-knee')
    ax.plot(x, z, zs=y)
    lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
