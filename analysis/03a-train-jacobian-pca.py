#!/usr/bin/env python

import climate
import itertools
import lmj.cubes
import lmj.pca
import os
import pandas as pd
import random

logging = climate.get_logger('03a-train-jac-pca')


@climate.annotate(
    root='load experiment data from this root',
    output='store pca and compressed data in this directory',
    pattern=('only load trials matching this pattern', 'option'),
)
def main(root, output, pattern='*'):
    if not os.path.isdir(output):
        os.makedirs(output)

    probes = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    trials = list(database.Experiment(root).trials_matching(pattern))

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


if __name__ == '__main__':
    climate.call(main)
