#!/usr/bin/env python

import climate
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


def svt(df, threshold=500, max_rmse=0.002, learning_rate=1, dropout_decay=0.1, window=10):
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
    dropout_decay : float, optional
        Weight linearly interpolated dropout frames with this decay rate.
        Defaults to 0.1.
    window : int, optional
        Model windows of this many consecutive frames. Defaults to 10.
    '''
    cols = marker_channel_columns(df, min_filled=0.01)
    data = df[cols]
    num_frames, num_channels = data.shape
    num_entries = num_frames * num_channels
    logging.info('SVT: missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - data.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * data.count().sum() / num_entries)

    linear = data.interpolate().ffill().bfill().values
    weights = np.zeros_like(linear)
    for i, c in enumerate(cols):
        weights[:, i] = np.exp(-dropout_decay * closest_observation(data[c]))

    t = np.concatenate([linear[i:num_frames-(window-i)] for i in range(window+1)], axis=1)
    w = np.concatenate([weights[i:num_frames-(window-i)] for i in range(window+1)], axis=1)

    logging.info('SVT: processing windowed data %s', t.shape)

    s = None
    x = y = np.zeros_like(t)
    rmse = max_rmse + 1
    i = 0
    while rmse > max_rmse:
        err = (t - x) * w
        y += learning_rate * err
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u * s, v)
        rmse = np.sqrt((err * err).mean().mean())
        logging.info('SVT %d: rmse %f using %d pcs %s',
                     i, rmse, len(s.nonzero()[0]), s[:10].astype('i'))
        i += 1

    df[cols] = np.concatenate([
        x[:, :num_channels],
        x[num_frames - window:, window * num_channels:],
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
        df[c] = scipy.signal.filtfilt(b, a, df[c])


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    frame_rate=('reindex frames to this rate', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    decay=('linear interpolation decay', 'option', None, float),
    freq=('lowpass filter at N Hz', 'option', None, float),
    window=('process windows of T frames', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, accuracy=0.002, threshold=200, decay=0.1, freq=10, window=5):
    for subject in database.Experiment(root).subjects:
        trials = [t for t in subject.trials if t.matches(pattern)]
        if not trials:
            continue
        for t in trials:
            t.load()
            t.reindex(frame_rate)
            t.mask_dropouts()
        # here we make a huge df containing all data for this subject.
        keys = [(t.block.key, t.key) for t in trials]
        df = svt(pd.concat([t.df for t in trials], keys=keys),
                 threshold=threshold,
                 max_rmse=accuracy,
                 learning_rate=1.5,
                 dropout_decay=decay,
                 window=window - 1,
        )
        for t in trials:
            t.df = df.ix[(t.block.key, t.key), :]
            lowpass(t.df, freq)
            t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
