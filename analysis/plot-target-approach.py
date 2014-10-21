import climate
import itertools
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='read subject data from this file tree',
    pattern=('plot data from files matching this pattern', 'option'),
    markers=('plot data for these mocap markers', 'option'),
    target_num=('plot data for this target', 'option', None, int),
    approach_sec=('plot variance for N sec prior to target acquisition', 'option', None, float),
)
def main(root,
         pattern='*/*/*circuit00*.csv.gz',
         markers='13-r-fing-index 25-l-fing-index 40-r-heel 37-r-knee',
         target_num=3,
         approach_sec=2):
    data = {(m, s): [] for m in markers.split() for s in range(12)}
    num_frames = int(100 * approach_sec)
    target_loc = []
    for trial in database.Experiment(root).trials_matching(pattern):
        move = trial.movement_to(target_num)
        move.normalize(frame_rate=100, order=1, accuracy=100)
        if not len(target_loc):
            target_loc = move.df.ix[:, 'target-x':'target-z']
        for marker, source in data:
            if all(move.df.source == source):
                df_ = move.df.loc[:, marker + '-x':marker + '-z'].reset_index(drop=True)
                df_.columns = list('xyz')
                data[marker, source].append(df_.iloc[-num_frames:, :])

    with plots.space() as ax:
        ax.scatter(target_loc['target-x'].iloc[0],
                   target_loc['target-z'].iloc[0],
                   target_loc['target-y'].iloc[0], marker='o', s=200, c='#111111', linewidth=0, alpha=0.7)
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
