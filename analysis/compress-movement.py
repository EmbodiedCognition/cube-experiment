#!/usr/bin/env python

import climate
import joblib
import lmj.pca
import os
import numpy as np

import database

logging = climate.get_logger('compress')

MARKERS = [
    'marker00-r-head-back',
    'marker01-r-head-front',
    'marker02-l-head-front',
    'marker03-l-head-back',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
    'marker11-r-fing-ring',
    'marker12-r-fing-middle',
    'marker13-r-fing-index',
    'marker14-r-mc-outer',
    'marker15-r-mc-inner',
    'marker16-r-thumb-base',
    'marker17-r-thumb-tip',
    'marker18-l-collar',
    'marker19-l-shoulder',
    'marker20-l-elbow',
    'marker21-l-wrist',
    'marker22-l-fing-pinky',
    'marker23-l-fing-ring',
    'marker24-l-fing-middle',
    'marker25-l-fing-index',
    'marker26-l-mc-outer',
    'marker27-l-mc-inner',
    'marker28-l-thumb-base',
    'marker29-l-thumb-tip',
    'marker30-abdomen',
    'marker31-sternum',
    'marker32-t3',
    'marker33-t9',
    'marker34-l-ilium',
    'marker35-r-ilium',
    'marker36-r-hip',
    'marker37-r-knee',
    'marker38-r-shin',
    'marker39-r-ankle',
    'marker40-r-heel',
    'marker41-r-mt-outer',
    'marker42-r-mt-inner',
    'marker43-l-hip',
    'marker44-l-knee',
    'marker45-l-shin',
    'marker46-l-ankle',
    'marker47-l-heel',
    'marker48-l-mt-outer',
    'marker49-l-mt-inner',
]

COLUMNS = ['{}-{}'.format(m, c) for m in MARKERS for c in 'xyz']


def compress(trial, target, variance=0.995):
    trial.load()
    trial.mask_fiddly_target_frames()
    trial.df = trial.df.dropna()

    body = database.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    pca = lmj.pca.PCA(filename=os.path.join(target, 'pca-body.npz'))
    for i, v in pca.encode(body.df[COLUMNS].values, variance).T:
        trial.df['body-pc{:02d}'] = pd.Series(v, index=trial.df.index)

    goal = database.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    for i, v in pca.encode(goal.df[COLUMNS].values, variance).T:
        trial.df['goal-pc{:02d}'] = pd.Series(v, index=trial.df.index)


def main(root, target, pattern='*'):
    trials = list(database.Experiment(root).trials_matching(pattern))
    for t in trials:
        t.load()
    # here we make a huge df containing all matching trial data.
    keys = [(t.block.key, t.key) for t in trials]

    df = pd.concat([t.df for t in trials], keys=keys)

    body = database.Trial(trial.parent, trial.basename)
    body.df = df.copy()
    body.make_body_relative()

    pca = lmj.pca.PCA()
    pca.fit(body.df[COLUMNS])
    for v in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999):
        print('{:.1f}%: {} body components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(target, 'pca-body-relative.npz'))

    goal = database.Trial(trial.parent, trial.basename)
    goal.df = df.copy()
    goal.make_target_relative()

    pca = lmj.pca.PCA()
    pca.fit(goals.df[COLUMNS])
    for v in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999):
        print('{:.1f}%: {} goal components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(target, 'pca-goal-relative.npz'))

    joblib.Parallel(-2)(joblib.delayed(compress)(t, target, variance) for t in trials)


if __name__ == '__main__':
    climate.call(main)
