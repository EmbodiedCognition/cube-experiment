#!/usr/bin/env python

import climate
import lmj.plot

import database
import plots


@climate.annotate(
    root='plot data from this experiment subjects',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*/*/*.csv.gz'):
    with lmj.plot.with_axes(spines=True) as ax:
        labeled = set()
        for trial in list(database.Experiment(root).trials_matching(pattern)):
            for t in range(12):
                mov = trial.movement_to(t)
                if not 0 < len(mov.df):# < 1000:
                    continue
                mov.drop_fiddly_target_frames()
                s = mov.distance_to_target()
                kw = dict(color=lmj.plot.COLOR11[t % 11], alpha=0.7, lw=2)
                if t not in labeled:
                    kw['label'] = t
                    labeled.add(t)
                ax.plot(range(len(s.values)), s.values[::-1], **kw)
        ax.set_xlabel('Frames Before Contact')
        ax.set_ylabel('Distance to Target')
        lmj.plot.legend()


if __name__ == '__main__':
    climate.call(main)
