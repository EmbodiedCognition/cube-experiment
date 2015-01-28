#!/usr/bin/env python

import climate
import joblib
import lmj.pca
import numpy as np

import database


def jac(trial, prefix):
    trial.load()
    cols = [c for c in trial.df.columns if c.startswith(prefix)]
    return trial.df[cols].values


@climate.annotate(
    root='load experiment data from this root',
    output='store pca and jacobians in this directory',
    pattern=('only load trials matching this pattern', 'option'),
    count=('only use this many jacobians for PCA', 'option', None, int),
    variance=('retain components to preserve this variance', 'option', None, float),
)
def main(root, output, pattern='*', count=10000, variance=0.99):
    trials = list(database.Experiment(root).trials_matching(pattern))
    proc = joblib.delayed(jac)
    for prefix in ('fwd', 'inv'):
        jacobia = []
        for jacs in joblib.Parallel(-1)(proc(t, 'jac-' + prefix) for t in trials):
            jacobia.extend(jacs)
        pca = lmj.pca.PCA()
        pca_file = os.path.join(output, 'pca-{}.npz'.format(prefix))
        if os.path.exists(pca_file):
            pca.load(pca_file)
        else:
            idx = np.arange(len(jacobia))
            np.random.shuffle(idx)
            pca.fit([jacobia[i] for i in idx[:count]])
            pca.save(pca_file)
            for v in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
                print('{:.1f}%: {} components'.format(100 * v, pca.num_components(v)))
        enc = pca.encode(jacobia, retain=variance)
        enc_file = os.path.join(output, 'jac-{}.npy'.format(prefix))
        logging.info('%s: saving %s %s', enc_file, enc.shape, enc.dtype)
        np.save(enc_file, enc)


if __name__ == '__main__':
    climate.call(main)
