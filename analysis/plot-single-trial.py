import climate
import lmj.plot

import source
import plots


@climate.annotate(
    subjects='plot data from these subjects',
    marker=('plot data for this mocap marker', 'option'),
    trial_num=('plot data for this trial', 'option', None, int),
)
def main(marker='r-fing-index', trial_num=0, *subjects):
    with plots.space() as ax:
        for i, subject in enumerate(subjects):
            subj = source.Subject(subject)
            for b in subj.blocks:
                trial = b.trials[trial_num]
                trial.load()
                df = trial.marker_trajectory(marker)
                ax.plot(df.x, df.z, zs=df.y, color=lmj.plot.COLOR11[i], alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
