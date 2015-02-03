#!/usr/bin/env python

import climate
import joblib
import numpy as np
import pandas as pd
import pypropack
import scipy.signal

import database

logging = climate.get_logger('fill')


def marker_channel_columns(df, min_filled=0):
    '''Return the names of columns that contain marker data.'''
    return [c for c in df.columns
            if c.startswith('marker') and
            c[-1] in 'xyz' and
            df[c].count() > min_filled * len(df)]


def closest_observation(series):
    '''Compute the distance (in frames) to the nearest non-dropout frame.

    Parameters
    ----------
    series : pd.Series
        A Series holding a single channel of mocap data.

    Returns
    -------
    closest : pd.Series
        An integer series containing, for each frame, the number of frames
        to the nearest non-dropout frame.
    '''
    drops = series.isnull()
    closest_l = [0]
    for d in drops:
        closest_l.append(1 + closest_l[-1] if d else 0)
    closest_r = [0]
    for d in drops[::-1]:
        closest_r.append(1 + closest_r[-1] if d else 0)
    return pd.Series(
        list(map(min, closest_l[1:], reversed(closest_r[1:]))),
        index=series.index)


def svt(df, threshold=None, tol=1e-3, learning_rate=1.5, window=10):
    '''Complete missing marker data using singular value thresholding.

    This method alters the given `df` in-place.

    Singular value thresholding is described in Cai, Candes, & Shen (2010),
    "A Singular Value Thresholding Algorithm for Matrix Completion" (see
    http://arxiv.org/pdf/0810.3286.pdf). The implementation here is rather
    naive but seems to get the job done for the types of mocap data that we
    gathered in the cube experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame for source data.
    threshold : float, optional
        Threshold for singular values. Defaults to the "natural" falling-off
        point in the singular value spectrum.
    tol : float, optional
        Continue the reconstruction process until reconstructed data is
        within this relative tolerance threshold. Defaults to 1e-3.
    learning_rate : float, optional
        Make adjustments to objective function with step size. Defaults to 1.5.
    window : int, optional
        Model windows of this many consecutive frames. Defaults to 10.
    '''
    cols = [c for c in marker_channel_columns(df, min_filled=0.01) if 'r-fing-pinky' not in c]
    data = df[cols]
    num_frames, num_channels = data.shape
    num_entries = num_frames * num_channels
    logging.info('SVT: missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * data.count().sum() / num_entries)

    filled = data.fillna(0).values
    weights = (~data.isnull()).values.astype(float)

    w = np.asfortranarray(np.concatenate([
        weights[i:num_frames-(window-i)] for i in range(window+1)], axis=1))
    t = w * np.asfortranarray(np.concatenate([
        filled[i:num_frames-(window-i)] for i in range(window+1)], axis=1))
    norm_t = np.linalg.norm(t)

    logging.info('SVT: processing windowed data %s', t.shape)

    # if the threshold is none, set it using the falloff point in the spectrum.
    if threshold is None:
        s = pd.Series(pypropack.svdp(
            t, k=20, kmax=200, compute_u=False, compute_v=False)[0])
        threshold = s[s.diff().diff().shift(-1).argmax()]
        logging.info('SVT: using threshold %.2f', threshold)

    x = None
    y = 0 + t
    i = topk = 0
    inck = 5
    while True:
        i += 1
        topk += inck
        u, s, v = pypropack.svdp(y, k=topk, kmax=10 * topk)
        while s[-1] > threshold:
            topk += inck
            u, s, v = pypropack.svdp(y, k=topk, kmax=10 * topk)
        s = np.clip(s - threshold, 0, np.inf)
        topk = len(s.nonzero()[0])
        x = np.dot(u * s, v)
        delta = t - w * x
        err = np.linalg.norm(delta) / norm_t
        logging.info('SVT %d: error %f using %d pcs %s',
                     i, err, topk, s[:10].astype('i'))
        if err < tol:
            break
        y += learning_rate * delta

    df[cols] = np.concatenate([
        x[:, :num_channels],
        x[num_frames - 2 * window:, window * num_channels:],
    ], axis=0)

    return df


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
    tol=('fit SVT with this error tolerance', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    freq=('lowpass filter at N Hz', 'option', None, float),
    window=('process windows of T frames', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, tol=0.02, threshold=None, freq=10, window=20):
    trials = list(database.Experiment(root).trials_matching(pattern))
    for t in trials:
        t.load()
        t.reindex(frame_rate)
        t.mask_dropouts()
    # here we make a huge df containing all matching trial data.
    keys = [(t.block.key, t.key) for t in trials]
    df = svt(pd.concat([t.df for t in trials], keys=keys),
             threshold=threshold, tol=tol, window=window - 1)
    for t in trials:
        t.df = df.ix[(t.block.key, t.key), :]
        lowpass(t.df, freq)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
