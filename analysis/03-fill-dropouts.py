#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import numpy as np
import pandas as pd
import scipy.signal

logging = climate.get_logger('fill')

CENTERS = [
    'marker34-l-ilium',
    'marker35-r-ilium',
    'marker36-r-hip',
    'marker37-r-knee',
]

def svt(dfs, tol=1e-4, threshold=None, window=5):
    '''Complete missing marker data using singular value thresholding.

    This method alters the given `df` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Data frames for source data. Each frame will be interpolated, and then
        the frames will be stacked into a single large frame to use during SVT.
        This stacked frame will then be split and returned.
    tol : float, optional
        Continue the reconstruction process until reconstructed data is within
        this relative tolerance threshold. Defaults to 1e-4.
    threshold : float, optional
        Threshold for singular values. Defaults to a value computed from the
        spectrum of singular values.
    window : int, optional
        Model windows of this many consecutive frames. Defaults to 5.

    Returns
    -------
    dfs : list of pd.DataFrame
        A list of data frames containing data with dropouts filled in.
    '''
    assert window > 1

    cols = [c for c in dfs[0].columns if c.startswith('marker') and c[-1] in 'xyz']
    pad = pd.DataFrame(float('nan'), index=list(range(window - 1)), columns=cols)
    chunks, keys = [pad], [-1]
    for i, df in enumerate(dfs):
        chunks.extend([df[cols], pad])
        keys.extend([i, -len(chunks)])
    data = pd.concat(chunks, axis=0, keys=keys)

    num_frames, num_channels = data.shape
    num_entries = num_frames * num_channels
    filled_ratio = data.count().sum() / num_entries
    logging.info('missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * filled_ratio)

    # if a column is completely missing, refuse to process the data.
    for c in cols:
        if data[c].count() == 0:
            raise ValueError('%s: no visible values!', c)

    # interpolate dropouts linearly.
    filled = data.interpolate().ffill().bfill()

    # shift the entire mocap recording by the location of the CENTERS (basically
    # the hip markers).
    center = pd.DataFrame(
        dict(x=filled[[m + '-x' for m in CENTERS]].mean(axis=1),
             y=filled[[m + '-y' for m in CENTERS]].mean(axis=1),
             z=filled[[m + '-z' for m in CENTERS]].mean(axis=1)))
    for c in cols:
        filled[c] -= center[c[-1]]

    filled = filled.values
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
    data_norm = (w * t * t).sum()
    logging.info('processing windowed data %s: norm %s', t.shape, data_norm)

    assert np.isfinite(data_norm)

    # if the threshold is none, set it using the falloff point in the spectrum.
    if threshold is None:
        s = pd.Series(np.linalg.svd(t, compute_uv=False))
        threshold = s[s.diff().diff().shift(-1).argmax()]
        logging.info('using threshold %.2f', threshold)

    i = 0
    err = 1e200
    x = y = np.zeros_like(t)
    while tol * data_norm < err:
        e = t - x
        err = (w * e * e).sum()
        y += e
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u * s, v)
        logging.info('%d: error %f (%d); mean %f; %s',
                     i, err / data_norm, len(s.nonzero()[0]), abs(e[w]).mean(),
                     np.percentile(abs(e[w]), [50, 90, 95, 99]).round(4))
        i += 1

    def avg(xs):
        return np.mean(list(xs), axis=0)

    parts = np.split(x, window, axis=1)
    w = window - 1

    # above, we created <window> duplicates of our data, each offset by 1 frame,
    # and stacked (along axis 1) into a big ol matrix. in effect, we have
    # <window> copies of each frame; here, we unpack these duplicates, remove
    # their offsets, take the average, and put them back in the linear frame.
    rows = pd.MultiIndex(levels=data.index.levels,
                         labels=[x[w:-w] for x in data.index.labels])
    data.loc[rows, cols] = avg(
        parts[j][w - j:num_frames - w - j] for j in range(window))

    # move our interpolated data back out to the world by restoring the
    # locations of the CENTERS.
    for c in cols:
        data.loc[:, c] += center.loc[:, c[-1]]

    # unstack the stacked data frame.
    for i, df in enumerate(dfs):
        df[cols] = data.loc[(i, ), cols]


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
    passes = 2  # filtfilt makes two passes over the data.
    correct = (2 ** (1 / passes) - 1) ** 0.25
    b, a = scipy.signal.butter(order / passes, (freq / correct) / nyquist)
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
def main(root, output, pattern='*', tol=1e-5, threshold=None, window=5, freq=None):
    for _, ts in lmj.cubes.Experiment(root).by_subject(pattern):
        ts = list(ts)
        for t in ts:
            t.load()
            t.mask_dropouts()
        try:
            svt([t.df for t in ts], tol, threshold, window)
        except Exception as e:
            logging.exception('error filling dropouts!')
            continue
        for i, t in enumerate(ts):
            if freq:
                lowpass(t.df, freq)
            t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
