#!/usr/bin/env python

import climate
import itertools
import joblib
import os
import pandas as pd
import random
import theanets

import database

logging = climate.get_logger('encode')


def encode(trial, output, network):
    trial.load()
    fwd = trial.df[[c for c in trial.columns if c.startswith('jac-fwd-pc')]]
    for i, v in enumerate(network.encode(fwd.values).T):
        trial.df['jac-fwd-enc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
    trial.save(trial.root.replace(trial.experiment.root, output))


g = climate.add_group('Jacobians')
g.add_argument('--root', help='load experiment data from this root')
g.add_argument('--output', help='store encoded data in this directory')
g.add_argument('--pattern', default='*', help='only load trials matching this pattern')
g.add_argument('--overcomplete', type=float, default=3, help='train Nx overcomplete dictionary')


def main(args):
    trials = list(database.Experiment(args.root).trials_matching(args.pattern))
    keys = [(t.block.key, t.key) for t in trials]

    # choose N trials per subject to train the dictionary.
    N = 5
    train_trials = []
    valid_trials = []
    for s, ts in itertools.groupby(trials, key=lambda t: t.subject.key):
        ts = list(ts)
        idx = list(range(len(ts)))
        #random.shuffle(idx)
        for i in idx[:N]:
            train_trials.append(ts[i])
            train_trials[-1].load()
        for i in idx[N:N+2]:
            valid_trials.append(ts[i])
            valid_trials[-1].load()

    cols = [c for c in train_trials[0].columns if c.startswith('jac-fwd-pc')]
    train = pd.concat([t.df for t in train_trials])[cols].dropna(axis=0).values.astype('f')
    valid = pd.concat([t.df for t in valid_trials])[cols].dropna(axis=0).values.astype('f')

    exp = theanets.Experiment(
        theanets.Autoencoder,
        layers=(train.shape[1], args.overcomplete * train.shape[1], train.shape[1]),
    )

    exp.train(train)

    exp.save(os.path.join(args.output, 'encoder.pkl.gz'))

    joblib.Parallel(3)(joblib.delayed(encode)(t, args.output, exp.network) for t in trials)


if __name__ == '__main__':
    climate.call(main)
