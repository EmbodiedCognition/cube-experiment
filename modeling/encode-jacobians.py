#!/usr/bin/env python

import climate
import gzip
import io
import joblib
import lmj.cubes
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.decomposition

logging = climate.get_logger('encode')


def load_jacobian(fn):
    df = pd.read_csv(fn, index_col='time').dropna()
    cols = [c for c in df.columns if c.startswith('jac')]
    return df[cols].astype('f')


def encode(pca, fn):
    df = load_jacobian(fn)
    cols = [c for c in df.columns if c.startswith('jac')]
    xf = pca.transform(df[cols].values)
    k = xf.shape[1]
    for c in cols:
        del df[c]
    for i in range(k):
        df['pc-{}'.format(i)] = xf[:, i]
    out = fn.replace('_jac', '_jac_pca{}'.format(k))
    s = io.StringIO()
    df.to_csv(s, index_label='time')
    with gzip.open(out, 'wb') as handle:
        handle.write(s.getvalue().encode('utf-8'))
    logging.info('saved %s to %s', df.shape, out)


def main(root):
    files = list(lmj.cubes.utils.matching(root, '*_jac.csv.gz'))
    np.random.shuffle(files)

    data = []
    cols = 0
    rows = 0
    for f in files:
        jac = load_jacobian(f)
        logging.info('%s: loaded %s', f, jac.shape)
        rows += jac.shape[0]
        cols = cols or jac.shape[1]
        data.append(jac.values.astype('f'))
        if rows > 3 * cols:
            break
    data = np.vstack(data)
    logging.info('computing pca on %s', data.shape)

    pca = sklearn.decomposition.PCA(n_components=0.99)
    pca.fit(data)

    cvar = ' '.join('{:.2f}'.format(100 * x) for x in
                    pca.explained_variance_ratio_.cumsum())
    logging.info('cumulative explained variance: %s', cvar)

    with open(os.path.join(root, 'jac-pca.pkl'), 'wb') as handle:
        pickle.dump(pca, handle)

    work = joblib.delayed(encode)
    joblib.Parallel(-1)(work(pca, f) for f in files)


if __name__ == '__main__':
    climate.call(main)
