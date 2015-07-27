import climate
import logging
import numpy as np
import os
import pickle
import theanets


def compress(source, k, key):
    logging.info('%s: looking for %s', source, key)

    filenames = sorted(
        fn for fn in os.listdir(source)
        if fn.endswith('{}.npy'.format(key)))
    arrays = [
        np.clip(np.load(os.path.join(source, fn), mmap_mode='r'), -1, 1)
        for fn in filenames]
    B, N = 128, arrays[0].shape[1]

    logging.info('loaded %s rows of %d-D data from %d files',
                 sum(len(a) for a in arrays), N, len(arrays))

    def batch():
        batch = np.zeros((B, N), 'f')
        for b in range(B):
            a = np.random.randint(len(arrays))
            j = np.random.randint(len(arrays[a]))
            batch[b] = arrays[a][j]
        return [batch]

    pca = theanets.Autoencoder([N, (k, 'linear'), (N, 'tied')])
    pca.train(batch, momentum=0.9)

    for arr, fn in zip(arrays, filenames):
        out = os.path.join(source, fn.replace(key, '{}_{}'.format(key, k)))
        karr = pca.encode(arr.astype('f'))
        logging.info('%s: saving %s', out, karr.shape)
        np.save(out, karr)

    out = os.path.join(source, 'pca_{}_{}.pkl'.format(key, k))
    pickle.dump(pca, open(out, 'wb'))


@climate.annotate(
    root='load data files from subject directories in this path',
    k=('compress to this many dimensions', 'option', None, int),
)
def main(root, k=400):
    for subject in sorted(os.listdir(root)):
        compress(os.path.join(root, subject), k, 'fjac')
        compress(os.path.join(root, subject), k, 'ijac')


if __name__ == '__main__':
    climate.call(main)
