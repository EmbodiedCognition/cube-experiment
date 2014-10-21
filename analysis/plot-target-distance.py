import climate
import itertools
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='plot data from this experiment subjects',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*block*/*circuit*.csv.gz'):
    with plots.plot() as ax:
        for trial in database.Experiment(root).trials_matching(pattern):
            for t in range(12):
                s = trial.movement_to(t).distance_to_target().interpolate().reset_index(drop=True)
                ax.plot(s.index, s.values, color=lmj.plot.COLOR11[t % 11])


if __name__ == '__main__':
    climate.call(main)
