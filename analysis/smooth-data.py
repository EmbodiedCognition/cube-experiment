#!/usr/bin/env python

import climate
import collections
import joblib
import numpy as np
import pandas as pd
import scipy.signal

import database

logging = climate.get_logger('smooth')


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


def svt(df, threshold=500, max_rmse=0.002, learning_rate=1, log_every=0):
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
    threshold : int, optional
        Threshold for singular values. Defaults to 500.
    max_rmse : float, optional
        Continue the reconstruction process until reconstructed data is
        below this RMS error compared with measured data. Defaults to 0.002.
    learning_rate : float, optional
        Make adjustments to objective function with step size. Defaults to 1.
    log_every : int, optional
        Number of SVT iterations between logging output. Defaults to 0,
        which only logs output at the start and finish of the SVT process.
    '''
    cols = marker_channel_columns(df, min_filled=0.01)
    data = df[cols]
    num_entries = data.shape[0] * data.shape[1]
    logging.info('SVT: missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 data.shape[0], data.shape[1], num_entries,
                 100 * data.count().sum() / num_entries)

    msg = 'SVT %d: rmse %f using %d singular values'
    log = lambda: logging.info(msg, i, rmse, len(s.nonzero()[0]))

    stdevs = max_rmse * closest_observation(data.iloc[:, 0])[:, None]
    linear = data.interpolate().ffill().bfill().values
    s = None
    x = y = np.zeros_like(linear)
    rmse = max_rmse + 1
    i = 0
    while rmse > max_rmse:
        err = linear - x + stdevs * np.random.randn(*x.shape)
        y += learning_rate * err
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u, np.dot(np.diag(s), v))
        rmse = np.sqrt((err * err).mean().mean())
        if log_every and i % log_every == 0: log()
        i += 1
    log()

    df[cols] = x


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
        df[c] = scipy.signal.filtfilt(b, a, df[c])


def smooth(args):
    t = args.trial
    t.load()
    t.reindex(args.frame_rate)
    t.mask_dropouts()
    svt(t.df,
        threshold=args.threshold,
        max_rmse=args.accuracy,
        log_every=0)
    lowpass(t.df, args.lowpass)
    t.save(t.root.replace(args.root, args.output))


Args = collections.namedtuple('Args', 'trial root output frame_rate accuracy threshold lowpass')

@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    frame_rate=('reindex frames to this rate', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    lowpass=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, frame_rate=100., accuracy=0.002, threshold=500, lowpass=10.):
    args = root, output, frame_rate, accuracy, threshold, lowpass
    trials = database.Experiment(root).trials_matching('*')
    proc = joblib.delayed(smooth)
    joblib.Parallel(-2)(proc(Args(t, *args) for t in trials))


if __name__ == '__main__':
    climate.call(main)
