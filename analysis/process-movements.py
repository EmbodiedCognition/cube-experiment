#!/usr/bin/env python

import climate
import itertools
import joblib
import lmj.pca
import os
import pandas as pd
import random

import database

logging = climate.get_logger('compress')


def compress(trial, output, variance=0.995):
    trial.load()
    trial.mask_fiddly_target_frames()
    #trial.df.dropna(thresh=len(COLUMNS), inplace=True)

    init = [c for c in trial.columns if c[:6] in ('source', 'target')]
    out = pd.DataFrame(trial.df[init], index=trial.df.index)

    def p(w):
        return os.path.join(output, 'pca-{}-relative.npz'.format(w))

    # encode body-relative data.
    body = database.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body_pcs = 0

    out['body-center-x'] = body['center-x']
    out['body-center-y'] = body['center-y']
    out['body-center-z'] = body['center-z']
    out['body-heading'] = body['heading']
    for c in body.columns:
        if c.endswith('-mean') or c.endswith('-std'):
            out[c] = body[c]

    pca = lmj.pca.PCA(filename=p('body'))
    for i, v in enumerate(pca.encode(body.df[body.marker_channel_columns].values, retain=variance).T):
        out['body-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
        body_pcs += 1

    # encode goal-relative data.
    goal = database.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal_pcs = 0

    out['goal-center-x'] = goal['center-x']
    out['goal-center-y'] = goal['center-y']
    out['goal-center-z'] = goal['center-z']
    out['goal-heading'] = goal['heading']

    pca = lmj.pca.PCA(filename=p('goal'))
    for i, v in enumerate(pca.encode(goal.df[goal.marker_channel_columns].values, retain=variance).T):
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
    probes = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    if variance not in probes:
        probes = sorted([variance] + probes)

    trials = list(database.Experiment(root).trials_matching(pattern))
    keys = [(t.block.key, t.key) for t in trials]

    # choose N trials per subject to compute the principal components.
    N = 3
    pca_trials = []
    for s, ts in itertools.groupby(trials, key=lambda t: t.subject.key):
        ts = list(ts)
        idx = list(range(len(ts)))
        random.shuffle(idx)
        for i in idx[:N]:
            pca_trials.append(ts[i])
            ts[i].load()

    body = database.Movement(pd.concat([t.df for t in pca_trials]))
    body.make_body_relative()

    pca = lmj.pca.PCA()
    pca.fit(body.df[body.marker_channel_columns])
    for v in probes:
        print('{:.1f}%: {} body components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-body-relative.npz'))

    goal = database.Movement(pd.concat([t.df for t in pca_trials]))
    goal.make_target_relative()

    pca = lmj.pca.PCA()
    pca.fit(goal.df[goal.marker_channel_columns])
    for v in probes:
        print('{:.1f}%: {} goal components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-goal-relative.npz'))

    joblib.Parallel(-1)(joblib.delayed(compress)(t, output, variance) for t in trials)


if __name__ == '__main__':
    climate.call(main)
