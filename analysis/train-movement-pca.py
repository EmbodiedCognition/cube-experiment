#!/usr/bin/env python

import climate
import itertools
import lmj.cubes
import lmj.pca
import os
import pandas as pd
import random

logging = climate.get_logger('train-pca')

@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern='process trials matching this pattern',
)
def main(root, output, pattern='*'):
    if not os.path.isdir(output):
        os.makedirs(output)

    probes = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    trials = list(lmj.cubes.Experiment(root).trials_matching(pattern))

    # choose N trials per subject to compute the principal components.
    N = 5
    pca_trials = []
    for s, ts in itertools.groupby(trials, key=lambda t: t.subject.key):
        ts = list(ts)
        idx = list(range(len(ts)))
        random.shuffle(idx)
        for i in idx[:N]:
            pca_trials.append(ts[i])
            ts[i].load()

    body = lmj.cubes.Movement(pd.concat([t.df for t in pca_trials]))
    body.make_body_relative()
    body.add_velocities()

    pca = lmj.pca.PCA()
    pca.fit(body.df[body.marker_channel_columns])
    for v in probes:
        print('{:.1f}%: {} body components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-body-relative.npz'))

    goal = lmj.cubes.Movement(pd.concat([t.df for t in pca_trials]))
    goal.make_target_relative()
    goal.add_velocities()

    pca = lmj.pca.PCA()
    pca.fit(goal.df[goal.marker_channel_columns])
    for v in probes:
        print('{:.1f}%: {} goal components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-goal-relative.npz'))


if __name__ == '__main__':
    climate.call(main)
