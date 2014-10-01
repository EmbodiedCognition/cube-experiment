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
def main(root, pattern='*/*block00/*circuit??.csv.gz', markers='r-fing-index l-fing-index r-heel r-knee', target_num=5, approach_sec=2):
    data = {(m, s): [] for m in markers.split() for s in range(12)}
    columns = num_frames = None
    for trial in experiment.Experiment(root).trials_matching(pattern):
        trial.realign(order=1)
        if columns is None:
            columns = {h: i for i, h in trial.marker_columns}
            num_frames = int(approach_sec * trial.approx_frame_rate)
        df = trial.movement_to(target_num)
        for marker, source in data:
            if all(df.source == source):
                col = columns[marker]
                df_ = df.iloc[-num_frames:, col:col+3].reset_index(drop=True)
                df_.columns = list('xyz')
                data[marker, source].append(df_)
    with plots.space() as ax:
        for i, (marker, keys) in enumerate(itertools.groupby(sorted(data), lambda x: x[0])):
            for j, (_, source) in enumerate(keys):
                for df in data[marker, source]:
                    ax.plot(np.asarray(df.x),
                            np.asarray(df.z),
                            zs=np.asarray(df.y),
                            color=lmj.plot.COLOR11[i],
                            alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
