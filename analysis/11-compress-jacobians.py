import climate
import logging
import numpy as np
import os

from sklearn.decomposition import PCA


def compress(source, k='mle', key='jac'):
    filenames = sorted(fn for fn in os.listdir(source)
                       if key in fn and fn.endswith('.npy'))
    logging.info('%s: found %d jacobians matching %s',
                 source, len(filenames), key)

    arrays = [np.load(os.path.join(source, fn)) for fn in filenames]
    for arr, fn in zip(arrays, filenames):
        if np.isnan(arr).any():
            logging.info('%s: %s contains %d nans!', fn, arr.shape, np.isnan(arr).sum())

    pca = PCA(n_components=k if k == 'mle' else int(k))
    pca.fit(np.vstack(arrays))

    for arr, fn in zip(arrays, filenames):
        out = os.path.join(source, fn.replace(key, '{}_{}'.format(key, k)))
        karr = pca.transform(arr)
        logging.info('%s: saving %s', out, karr.shape)
        np.save(out, karr)

    out = os.path.join(source, 'pca_{}_{}.pkl'.format(key, k))
    pickle.dump(open(out, 'wb'), pca)


def main(root, k='mle'):
    for subject in sorted(os.listdir(root)):
        compress(os.path.join(root, subject), k, 'jac')


if __name__ == '__main__':
    climate.call(main)
