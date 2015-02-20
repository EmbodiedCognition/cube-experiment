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


def fill(df, tol=1e-4, window=5, gamma=1, rank=None):
    '''Complete missing marker data using matrix completion.

    This method alters the given `df` in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame for source data.
    tol : float, optional
        Continue the reconstruction process until reconstructed data is within
        this relative tolerance threshold. Defaults to 1e-4.
    window : int, optional
        Model windows of this many consecutive frames. Defaults to 5.
    gamma : float, optional
        Use this parameter to regularize the low-rank entries on the factored
        data matrix. Defaults to 1.
    rank : int, optional
        Approximate the data using a matrix of this rank. Defaults to the
        number of SVD components required to match the filled percentage of the
        data.
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
    w = np.concatenate([weights[i:num_frames-(window-1-i)] for i in range(window)], axis=1).astype(np.int8)
    t = np.concatenate([filled[i:num_frames-(window-1-i)] for i in range(window)], axis=1).astype(np.float32)
    logging.info('processing windowed data %s', t.shape)
    data_norm = (w * t * t).sum()

    # compute one svd to initialize the optimization parameters below.
    u, s, v = scipy.linalg.svd(t, full_matrices=False)
    cdf = (s / s.sum()).cumsum()
    rank = rank or cdf.searchsorted(filled_ratio)
    logging.info('reconstructing with rank %d (%d%%) %s',
                 rank, 100 * cdf[rank],
                 [(f, cdf.searchsorted(f)) for f in (0.5, 0.8, 0.9, 0.99)])

    # optimization parameters
    gamma = TT.cast(gamma, 'float32')
    learning_rate = TT.cast(0.001, 'float32')
    momentum = TT.cast(0.9, 'float32')
    max_norm = TT.cast(10, 'float32')
    one = TT.cast(1, 'float32')

    # data variables.
    data = theano.shared(t.astype('f'))
    mask = theano.shared(w.astype('f'))

    # model parameters.
    ss = np.sqrt(np.clip(s[:rank] - s[int(np.sqrt(rank))], 0, 1e300))
    u = theano.shared((u[:, :rank] * ss).astype('f'), name='u')
    vu = theano.shared(np.zeros_like(u.get_value()), name='vu')
    v = theano.shared((ss[:, None] * v[:rank]).astype('f'), name='v')
    vv = theano.shared(np.zeros_like(v.get_value()), name='vv')

    # symbolic computations, including sgd with momentum.
    def clip(g):
        return g * TT.minimum(one, max_norm / TT.sqrt((g * g).sum()))
    err = data - TT.dot(u, v)
    sqerr = (mask * err * err).sum()
    loss = sqerr + gamma * ((u * u).sum() + (v * v).sum())
    gu, gv = TT.grad(loss, [u, v])
    velu = momentum * vu - learning_rate * clip(gu)
    velv = momentum * vv - learning_rate * clip(gv)
    fu = theano.function([], [sqerr, gu], updates=[(vu, velu), (u, u + velu)])
    fv = theano.function([], [sqerr, gv], updates=[(vv, velv), (v, v + velv)])

    i = 0
    err = 1e200
    prev_err = 1e300
    while tol * data_norm < err < prev_err:
        prev_err = err
        err, gu = fu()
        err, gv = fv()
        if not i % 10:
            r = w * (t - np.dot(u.get_value(), v.get_value()))
            logging.info('%d: error %f (sanity: %f; %f mean; %s) gu %.1f gv %.1f',
                         i, err / data_norm,
                         (r * r).sum() / data_norm,
                         abs(r).mean(),
                         np.percentile(abs(r), [50, 90, 95, 99]).round(4),
                         np.sqrt((gu * gu).sum()),
                         np.sqrt((gv * gv).sum()))
        i += 1

    def avg(xs):
        return np.mean(list(xs), axis=0)

    def idx(a, b=None):
        return slice(df.index[a], df.index[b] if b else df.index[a])

    parts = np.split(np.dot(u.get_value(), v.get_value()), window, axis=1)
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
    window=('process windows of T frames', 'option', None, int),
    rank=('reconstruct data using matrix of this rank', 'option', None, int),
    freq=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, pattern='*', tol=0.0001, window=5, rank=400, freq=20):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        t.mask_dropouts()
        try:
            fill(t.df, tol=tol, window=window, rank=rank)
        except Exception as e:
            logging.exception('error filling dropouts!')
        if freq:
            lowpass(t.df, freq)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
