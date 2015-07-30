import climate
import glob
import gzip
import io
import lmj.cubes
import logging
import numpy as np
import os
import pandas as pd
import pickle
import theanets


def compress(source, k, activation, **kwargs):
    fns = sorted(glob.glob(os.path.join(source, '*', '*_jac.csv.gz')))
    logging.info('%s: found %d jacobians', source, len(fns))

    # the clipping operation affects about 2% of jacobian values.
    dfs = [np.clip(pd.read_csv(fn, index_col='time').dropna(), -10, 10)
           for fn in fns]

    B, N = 128, dfs[0].shape[1]

    logging.info('loaded %s rows of %d-D data from %d files',
                 sum(len(df) for df in dfs), N, len(dfs))

    def batch():
        batch = np.zeros((B, N), 'f')
        for b in range(B):
            a = np.random.randint(len(dfs))
            batch[b] = dfs[a].iloc[np.random.randint(len(dfs[a])), :]
        return [batch]

    pca = theanets.Autoencoder([N, (k, activation), (N, 'tied')])
    pca.train(batch, **kwargs)

    key = '{}_k{}'.format(activation, k)
    if 'hidden_l1' in kwargs:
        key += '_s{hidden_l1:.4f}'.format(**kwargs)

    for df, fn in zip(dfs, fns):
        df = pd.DataFrame(pca.encode(df.values.astype('f')), index=df.index)
        s = io.StringIO()
        df.to_csv(s, index_label='time')
        out = fn.replace('_jac', '_jac_' + key)
        with gzip.open(out, 'wb') as handle:
            handle.write(s.getvalue().encode('utf-8'))
        logging.info('%s: saved %s', out, df.shape)

    out = os.path.join(source, 'pca_{}.pkl'.format(key))
    pickle.dump(pca, open(out, 'wb'))


@climate.annotate(
    root='load data files from subject directories in this path',
    k=('compress to this many dimensions', 'option', None, int),
    activation=('use this activation function', 'option'),
)
def main(root, k=1000, activation='relu'):
    for subject in lmj.cubes.Experiment(root).subjects:
        compress(subject.root, k, activation,
                 momentum=0.9,
                 hidden_l1=0.01,
                 weight_l1=0.01,
                 monitors={'hid1:out': (0.01, 0.1, 1, 10)})


if __name__ == '__main__':
    climate.call(main)
