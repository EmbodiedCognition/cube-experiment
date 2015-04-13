#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.pca
import os
import pandas as pd

logging = climate.get_logger('03a-train-jac-pca')

PROBES = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]

@climate.annotate(
    root='load experiment data from this root',
    output='store pca and compressed data in this directory',
    pattern=('only load trials matching this pattern', 'option'),
    n=('choose N trials per subject to compute the PCs', 'option', None, int),
)
def main(root, output, pattern='*', n=3):
    if not os.path.isdir(output):
        os.makedirs(output)

    df = pd.concat([t.df for t in lmj.cubes.Experiment(root).load_sample(pattern, n)])

    fwd = df[[c for c in df.columns if c.startswith('jac-g')]]
    n = fwd.shape[0] * fwd.shape[1]
    logging.info('forward jacobian: %s x %s', fwd.shape[0], fwd.shape[1])
    for thresh in (-3, -2, -1, 0, 1):
        c = (abs(fwd) < (10 ** thresh)).sum().sum()
        logging.info('components < 1e%+d %10d %.1f%%', thresh, c, 100 * c / n)

    pca = lmj.pca.PCA()
    pca.fit(fwd)
    for v in PROBES:
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
    for v in PROBES:
        logging.info('%.1f%%: %d inv components', 100 * v, pca.num_components(v))
    pca.save(os.path.join(output, 'pca-jac-inv.npz'))


if __name__ == '__main__':
    climate.call(main)
