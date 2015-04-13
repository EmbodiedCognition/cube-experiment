#!/usr/bin/env python

import climate
import itertools
import joblib
import lmj.cubes
import lmj.pca
import os
import pandas as pd
import random

logging = climate.get_logger('02b-compute-jac')


def compute(trial, output, variance=0.995):
    trial.load()
    trial.mask_fiddly_target_frames()
    #trial.df.dropna(thresh=len(COLUMNS), inplace=True)

    init = [c for c in trial.columns
            if c.startswith('source') or c.startswith('target')]
    out = pd.DataFrame(trial.df[init], index=trial.df.index)

    def p(w):
        return os.path.join(output, 'pca-{}-relative.npz'.format(w))

    # encode body-relative data.
    body = lmj.cubes.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    stats = body.make_body_relative()
    body.add_velocities()
    body_pcs = 0

    out['body-center-x'] = body.df['center-x']
    out['body-center-y'] = body.df['center-y']
    out['body-center-z'] = body.df['center-z']
    out['body-heading'] = body.df['heading']

    pca = lmj.pca.PCA(filename=p('body'))
    values = body.df[body.marker_channel_columns].values
    for variance in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
        encoded = pca.encode(values, retain=variance)
        recons = pd.DataFrame(pca.decode(encoded, retain=variance),
                              columns=body.marker_channel_columns, index=body.df.index)
        err = body.df[body.marker_position_columns] - recons[body.marker_position_columns] * stats.sigma + stats.mu
        logging.info('variance %f: reconstruction error: %f', variance, (err * err).mean().mean())
    return

    for i, v in enumerate(encoded.T):
        out['body-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
        body_pcs += 1

    # encode goal-relative data.
    goal = lmj.cubes.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal.add_velocities()
    goal_pcs = 0

    out['goal-center-x'] = goal.df['center-x']
    out['goal-center-y'] = goal.df['center-y']
    out['goal-center-z'] = goal.df['center-z']
    out['goal-heading'] = goal.df['heading']

    pca = lmj.pca.PCA(filename=p('goal'))
    values = goal.df[goal.marker_channel_columns].values
    for i, v in enumerate(pca.encode(values, retain=variance).T):
        out['goal-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
        goal_pcs += 1

    # add columns for the jacobian.
    for bpc in range(body_pcs):
        db = out['body-pc{:02d}'.format(bpc)].diff()
        db[db == 0] = float('nan')
        for gpc in range(goal_pcs):
            dg = out['goal-pc{:02d}'.format(gpc)].diff()
            dg[dg == 0] = float('nan')
            out['jac-g{:02d}/b{:02d}'.format(gpc, bpc)] = dg / db
            out['jac-b{:02d}/g{:02d}'.format(bpc, gpc)] = db / dg

    trial.df = out[sorted(out.columns)]
    trial.save(trial.root.replace(trial.experiment.root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern='process trials matching this pattern',
    variance=('retain this fraction of variance', 'option', None, float),
)
def main(root, output, pattern='*', variance=0.99):
    func = joblib.delayed(compute)
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    #joblib.Parallel(-1)(func(t, output, variance) for t in trials)
    compute(list(trials)[0], output, variance)


if __name__ == '__main__':
    climate.call(main)
