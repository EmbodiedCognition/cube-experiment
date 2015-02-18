#!/usr/bin/env python

import climate
import lmj.cubes
import numpy as np
import pandas as pd
import scipy.signal
import theano
import theano.tensor as TT

logging = climate.get_logger('fill')

# this is the set of markers that gets included in our output.
MARKERS = [
    'marker00-r-head-back',
    'marker01-r-head-front',
    'marker02-l-head-front',
    'marker03-l-head-back',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
    'marker12-r-fing-middle',
    'marker13-r-fing-index',
    'marker14-r-mc-outer',
    'marker15-r-mc-inner',
    'marker16-r-thumb-base',
    'marker17-r-thumb-tip',
    'marker18-l-collar',
    'marker19-l-shoulder',
    'marker20-l-elbow',
    'marker21-l-wrist',
    'marker24-l-fing-middle',
    'marker25-l-fing-index',
    'marker26-l-mc-outer',
    'marker27-l-mc-inner',
    'marker28-l-thumb-base',
    'marker29-l-thumb-tip',
    'marker30-abdomen',
    'marker31-sternum',
    'marker32-t3',
    'marker33-t9',
    'marker34-l-ilium',
    'marker35-r-ilium',
    'marker36-r-hip',
    'marker37-r-knee',
    'marker38-r-shin',
    'marker39-r-ankle',
    'marker40-r-heel',
    'marker41-r-mt-outer',
    'marker42-r-mt-inner',
    'marker43-l-hip',
    'marker44-l-knee',
    'marker45-l-shin',
    'marker46-l-ankle',
    'marker47-l-heel',
    'marker48-l-mt-outer',
    'marker49-l-mt-inner',
]


def marker_channel_columns(df):
    '''Return the names of columns that contain marker data.'''
    return [c for c in df.columns if c[:-2] in MARKERS and c[-1] in 'xyz']


def fill(df, tol=1e-4, window=20, gamma=1):
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
        Model windows of this many consecutive frames. Defaults to 20.
    gamma : float, optional
        Use this parameter to regularize the low-rank entries on the factored
        data matrix. Defaults to 1.
    '''
    cols = marker_channel_columns(df)
    data = df[cols]
    num_frames, num_channels = data.shape
    num_entries = num_frames * num_channels
    filled_ratio = data.count().sum() / num_entries
    logging.info('missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * filled_ratio)

    filled = data.fillna(0).values
    weights = (~data.isnull()).values.astype(float)

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
    t = np.concatenate([filled[i:num_frames-(window-1-i)] for i in range(window)], axis=1).astype('f')
    logging.info('processing windowed data %s', t.shape)

    rank = 600
    s = scipy.linalg.svd(t, compute_uv=False)
    cdf = (s / s.sum()).cumsum()
    logging.info('reconstructing with rank %d (%d%%) %s',
                 rank, 100 * cdf[rank],
                 [(f, cdf.searchsorted(f)) for f in (0.5, 0.8, 0.9, 0.99)])

    # optimization parameters
    gamma = TT.cast(gamma, 'float32')
    learning_rate = TT.cast(0.0001, 'float32')
    momentum = TT.cast(0.94, 'float32')
    max_norm = TT.cast(100000, 'float32')
    one = TT.cast(1, 'float32')

    # symbolic variables.
    data = TT.matrix('data')
    mask = TT.bmatrix('mask')

    # model parameters.
    u = theano.shared(np.random.randn(t.shape[0], rank).astype('f'), name='u')
    v = theano.shared(np.random.randn(rank, t.shape[1]).astype('f'), name='v')

    # symbolic computations, including sgd with momentum.
    err = data - TT.dot(u, v)
    sqerr = (mask * err * err).sum()
    sqdat = (mask * data * data).sum()
    loss = sqerr + gamma * ((u * u).sum() + (v * v).sum())
    updates = []
    for param, grad in zip([u, v], TT.grad(loss, [u, v])):
        g = grad * TT.minimum(one, max_norm / TT.sqrt((grad * grad).sum()))
        v = theano.shared(np.zeros_like(param.get_value()), name='v' + param.name)
        vnew = momentum * v - learning_rate * g
        updates.append((v, vnew))
        updates.append((param, param + vnew))
    f = theano.function([data, mask], [sqerr, sqerr / sqdat], updates=updates)

    i = 0
    n = w.sum()
    rerr = tol + 1
    while rerr > tol:
        aerr, rerr = f(t, w)
        if not i % 10:
            logging.info('%d: abs error %f, relative error %f',
                         i, np.sqrt(aerr / n), rerr)
        i += 1
    del t
    del w

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
    for c in marker_channel_columns(df):
        df.loc[:, c] = scipy.signal.filtfilt(b, a, df[c])


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    frame_rate=('reindex frames to this rate', 'option', None, float),
    tol=('fill dropouts with this error tolerance', 'option', None, float),
    window=('process windows of T frames', 'option', None, int),
    freq=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, tol=0.0001, window=20, freq=20):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        t.mask_dropouts()
        t.reindex(frame_rate)
        for c in t.columns:
            if c.startswith('marker') and c[:-2] not in MARKERS:
                del t.df[c]
        try:
            fill(t.df, tol=tol, window=window)
        except Exception as e:
            logging.exception('error filling dropouts!')
        lowpass(t.df, freq)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
