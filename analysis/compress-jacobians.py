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


def compress(trial, output, variance=0.999):
    trial.load()

    init = [c for c in trial.columns if not c.startswith('jac-')]
    out = pd.DataFrame(trial.df[init], index=trial.df.index)

    def p(w):
        return os.path.join(output, 'pca-jac-{}.npz'.format(w))

    # encode forward jacobian.
    fwd = trial.df[[c for c in trial.columns if c.startswith('jac-g')]]
    pca = lmj.pca.PCA(filename=p('fwd'))
    for i, v in enumerate(pca.encode(fwd.values, retain=variance).T):
        out['jac-fwd-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)

    # encode forward jacobian.
    inv = trial.df[[c for c in trial.columns if c.startswith('jac-b')]]
    pca = lmj.pca.PCA(filename=p('inv'))
    for i, v in enumerate(pca.encode(inv.values, retain=variance).T):
        out['jac-inv-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)

    trial.df = out[sorted(out.columns)]
    trial.save(trial.root.replace(trial.experiment.root, output))


@climate.annotate(
    root='load experiment data from this root',
    output='store pca and compressed data in this directory',
    pattern=('only load trials matching this pattern', 'option'),
    variance=('retain components to preserve this variance', 'option', None, float),
)
def main(root, output, pattern='*', variance=0.999):
    probes = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    if variance not in probes:
        probes = sorted([variance] + probes)

    trials = list(database.Experiment(root).trials_matching(pattern))
    keys = [(t.block.key, t.key) for t in trials]

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

    df = pd.concat([t.df for t in pca_trials])

    fwd = df[[c for c in df.columns if c.startswith('jac-g')]]
    n = fwd.shape[0] * fwd.shape[1]
    logging.info('forward jacobian: %s x %s', fwd.shape[0], fwd.shape[1])
    for thresh in (-3, -2, -1, 0, 1):
        c = (abs(fwd) < (10 ** thresh)).sum().sum()
        logging.info('components < 1e%+d %10d %.1f%%', thresh, c, 100 * c / n)
    pca = lmj.pca.PCA()
    pca.fit(fwd)
    for v in probes:
        logging.info('%.1f%%: %d fwd components', 100 * v, pca.num_components(v))
    pca.save(os.path.join(output, 'pca-jac-fwd.npz'))
    del fwd

    inv = df[[c for c in df.columns if c.startswith('jac-b')]]
    n = inv.shape[0] * inv.shape[1]
    logging.info('forward jacobian: %s x %s', inv.shape[0], inv.shape[1])
    for thresh in (-3, -2, -1, 0, 1):
        c = (abs(inv) < (10 ** thresh)).sum().sum()
        logging.info('components < 1e%+d %10d %.1f%%', thresh, c, 100 * c / n)
    pca = lmj.pca.PCA()
    pca.fit(inv)
    for v in probes:
        logging.info('%.1f%%: %d inv components', 100 * v, pca.num_components(v))
    pca.save(os.path.join(output, 'pca-jac-inv.npz'))
    del inv

    del df

    return [compress(t, output, variance) for t in trials]
    joblib.Parallel(-1)(joblib.delayed(compress)(t, output, variance) for t in trials)


if __name__ == '__main__':
    climate.call(main)
