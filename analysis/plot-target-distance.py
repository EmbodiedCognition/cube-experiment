import climate
import itertools
import lmj.plot
import numpy as np

import source as experiment
import plots


@climate.annotate(
    root='plot data from this experiment subjects',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot data for these mocap markers', 'option'),
    target_num=('plot data for this target', 'option', None, int),
    approach_sec=('plot variance for N sec prior to target acquisition', 'option', None, float),
)
def main(root, pattern='*/*block*/*circuit*.csv.gz', markers='r-fing-index l-fing-index r-heel r-knee', target_num=5, approach_sec=2):
    with plots.plot() as ax:
        for i, trial in enumerate(experiment.Experiment(root).trials_matching(pattern)):
            for t in range(11):
                s = trial.movement_to(t).distance_to_target().interpolate().reset_index(drop=True)
                ax.plot(s.index, s.values, color=lmj.plot.COLOR11[t])


if __name__ == '__main__':
    climate.call(main)
