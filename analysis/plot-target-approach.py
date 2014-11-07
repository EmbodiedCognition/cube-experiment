import climate
import itertools
import lmj.plot
import numpy as np
import pandas as pd

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
         pattern='*/*/*trial00*.csv.gz',
         markers='r-fing-index l-fing-index r-head-front r-heel r-knee',
         target_num=3,
         approach_sec=2):
    data = {(m, s): [] for m in markers.split() for s in range(12)}
    num_frames = int(100 * approach_sec)
    targets = None
    for trial in database.Experiment(root).trials_matching(pattern):
        if targets is None:
            targets = trial
        move = trial.movement_to(target_num)
        move.df = pd.DataFrame(np.asarray(move.df),
                               columns=move.df.columns,
                               index=np.arange(1 - len(move.df), 1))
        for marker, source in data:
            if all(move.df.source == source):
                data[marker, source].append(move.trajectory(marker))
    with plots.space() as ax:
        plots.show_cubes(ax, targets, target_num=target_num)
        for i, (marker, keys) in enumerate(itertools.groupby(sorted(data), lambda x: x[0])):
            for j, (_, source) in enumerate(keys):
                dfs = data[marker, source]
                if not dfs:
                    continue
                for df in dfs:
                    ax.plot(np.asarray(df[-num_frames:].x),
                            np.asarray(df[-num_frames:].z),
                            zs=np.asarray(df[-num_frames:].y),
                            color=lmj.plot.COLOR11[i],
                            alpha=0.3)
                merged = pd.concat(dfs, axis=1, keys=list(range(len(dfs)))).groupby(axis=1, level=1)
                mu = merged.mean()
                sigma = merged.std()
                for t in np.linspace(0, num_frames, 5).astype(int):
                    try:
                        x, y, z = plots.ellipsoid([mu.x[-t], mu.y[-t], mu.z[-t]],
                                                  [sigma.x[-t], sigma.y[-t], sigma.z[-t]])
                    except:
                        continue
                    ax.plot_wireframe(x, z, y, color=lmj.plot.COLOR11[i], alpha=0.7)


if __name__ == '__main__':
    climate.call(main)
