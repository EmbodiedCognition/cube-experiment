#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.signal
import theano
import theano.tensor as TT

logging = climate.get_logger('fill')


def svt(df, tol=1e-4, threshold=None, window=5):
    '''Complete missing marker data using singular value thresholding.

    This method alters the given `df` in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame for source data.
    tol : float, optional
        Continue the reconstruction process until reconstructed data is within
        this relative tolerance threshold. Defaults to 1e-4.
    threshold : float, optional
        Threshold for singular values. Defaults to a value computed from the
        spectrum of singular values.
    window : int, optional
        Model windows of this many consecutive frames. Defaults to 5.
    '''
    cols = [c for c in df.columns if c.startswith('marker') and c[-1] in 'xyz']
    data = df[cols]
    num_frames, num_channels = data.shape
    num_entries = num_frames * num_channels
    filled_ratio = data.count().sum() / num_entries
    logging.info('missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * filled_ratio)

    filled = data.fillna(0).values
    weights = (~data.isnull()).values

    # here we create windows of consecutive data frames, all stacked together
    # along axis 1. for example, with 10 2-dimensional frames, we can stack them
    # into windows of length 4 as follows:
    #
    #     data   windows
    # 0   A B    A B C D E F G H
    # 1   C D    C D E F G H I J
    # 2   E F    E F G H I J K L
    # 3   G H    G H I J K L M N
    # 4   I J    I J K L M N O P
    # 5   K L    K L M N O P Q R
    # 6   M N    M N O P Q R S T
    # 7   O P
    # 8   Q R
    # 9   S T
    #
    # this is more or less like a convolution with a sliding rectangular window.
    # this stacked data matrix is the one we'll want to fill in the SVT process
    # below.
    w = np.concatenate([weights[i:num_frames-(window-1-i)] for i in range(window)], axis=1)
    t = np.concatenate([filled[i:num_frames-(window-1-i)] for i in range(window)], axis=1)
    logging.info('processing windowed data %s', t.shape)
    data_norm = (w * t * t).sum()

    # if the threshold is none, set it using the falloff point in the spectrum.
    if threshold is None:
        s = pd.Series(np.linalg.svd(t, compute_uv=False))
        threshold = s[s.diff().diff().shift(-1).argmax()]
        logging.info('using threshold %.2f', threshold)

    i = 0
    err = 1e200
    x = y = np.zeros_like(t)
    while tol * data_norm < err:
        r = (t - x) * w
        y += r
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u * s, v)
        err = (r * r).sum()
        logging.info('%d: error %f (%d); mean %f; %s',
                     i, err / data_norm, len(s.nonzero()[0]), abs(r).mean(),
                     np.percentile(abs(r), [50, 90, 95, 99]).round(4))
        i += 1

    def avg(xs):
        return np.mean(list(xs), axis=0)

    def idx(a, b=None):
        return slice(df.index[a], df.index[b] if b else df.index[a])

    parts = np.split(x, window, axis=1)
    w = window - 1
    f = num_frames - 1

    # super confusing bit! above, we created <window> duplicates of our data,
    # each offset by 1 frame, and stacked (along axis 1) into a big ol matrix.
    # here we unpack these duplicates, remove their offsets, take the average of
    # the appropriate values, and put them back in the data frame.
    #
    # this is particularly tricky because the first and last <window> frames
    # only appear a small number of times in the overall matrix. but the
    # indexing shenanigans below seem to do the trick.
    df.loc[idx(w, f - w), cols] = avg(parts[j][w - j:f + 1 - w - j] for j in range(window))
    for i in range(window):
        df.loc[idx(i), cols] = avg(parts[i - j][j] for j in range(i, -1, -1))
        df.loc[idx(f - i), cols] = avg(parts[w - (i - j + 1)][-j] for j in range(i+1, 0, -1))


def lowpass(df, freq=10., order=4):
    '''Filter marker data using a butterworth low-pass filter.

    This method alters the data in `df` in-place.

    Parameters
    ----------
    freq : float, optional
        Use a butterworth filter with this cutoff frequency. Defaults to
        10Hz.
    order : int, optional
        Order of the butterworth filter. Defaults to 4.
    '''
    nyquist = 1 / (2 * pd.Series(df.index).diff().mean())
    assert 0 < freq < nyquist
    b, a = scipy.signal.butter(order, freq / nyquist)
    for c in df.columns:
        if c.startswith('marker') and c[-1] in 'xyz':
            df.loc[:, c] = scipy.signal.filtfilt(b, a, df[c])


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    tol=('fill dropouts with this error tolerance', 'option', None, float),
    threshold=('SVT threshold for singular values', 'option', None, float),
    window=('process windows of T frames', 'option', None, int),
    freq=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, pattern='*', tol=0.0001, threshold=None, window=5, freq=20):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        t.mask_dropouts()
        try:
            svt(t.df, tol, threshold, window)
        except Exception as e:
            logging.exception('error filling dropouts!')
        if freq:
            lowpass(t.df, freq)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
