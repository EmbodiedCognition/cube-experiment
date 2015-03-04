#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.pca
import numpy as np
import os
import pandas as pd

logging = climate.get_logger('02a-train-move-pca')

PROBES = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]

@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern=('process trials matching this pattern', 'option'),
    n=('choose N trials per subject to compute the PCs', 'option', None, int),
)
def main(root, output, pattern='*', n=5):
    if not os.path.isdir(output):
        os.makedirs(output)

    trials = lmj.cubes.Experiment(root).load_sample(pattern, n)

    body = lmj.cubes.Movement(pd.concat([t.df for t in trials]))
    stats = body.make_body_relative()
    body.add_velocities()

    pca = lmj.pca.PCA()
    pca.fit(body.df[body.marker_channel_columns])
    for v in PROBES:
        print('{:.1f}%: {} body components'.format(100 * v, pca.num_components(v)))
    stats.to_csv(os.path.join(output, 'zscores-body-relative.csv'))
    pca.save(os.path.join(output, 'pca-body-relative.npz'))

    goal = lmj.cubes.Movement(pd.concat([t.df for t in trials]))
    goal.make_target_relative()
    goal.add_velocities()

    pca = lmj.pca.PCA()
    pca.fit(goal.df[goal.marker_channel_columns])
    for v in PROBES:
        print('{:.1f}%: {} goal components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-goal-relative.npz'))


if __name__ == '__main__':
    climate.call(main)
