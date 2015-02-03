#!/usr/bin/env python

import climate
import joblib
import numpy as np
import pandas as pd
import pypropack
import scipy.signal

import database

logging = climate.get_logger('fill')

MARKERS = [
    'marker00-r-head-back',
    'marker01-r-head-front',
    'marker02-l-head-front',
    'marker03-l-head-back',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
    'marker13-r-fing-index',
    'marker14-r-mc-outer',
    'marker15-r-mc-inner',
    'marker16-r-thumb-base',
    'marker17-r-thumb-tip',
    'marker18-l-collar',
    'marker19-l-shoulder',
    'marker20-l-elbow',
    'marker21-l-wrist',
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

def marker_channel_columns(df, min_filled=0):
    '''Return the names of columns that contain marker data.'''
    return [c for c in df.columns if c[:-2] in MARKERS and c[-1] in 'xyz']


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
    cols = marker_channel_columns(df)
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
        weights[i:num_frames-(window-i)] for i in range(window+1)], axis=1).astype('f'))
    t = np.asfortranarray(np.concatenate([
        filled[i:num_frames-(window-i)] for i in range(window+1)], axis=1).astype('f'))
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
        delta = w * (t - x)
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
    window=('process windows of T frames', 'option', None, int),
    freq=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, tol=0.02, window=30, freq=10):
    trials = list(database.Experiment(root).trials_matching(pattern))
    for t in trials:
        t.load()
        t.reindex(frame_rate)
        t.mask_dropouts()
        for c in t.columns:
            if c.startswith('marker') and c[:-2] not in MARKERS:
                del t.df[c]

    # here we make a huge df containing all matching trial data.
    keys = [(t.block.key, t.key) for t in trials]
    df = pd.concat([t.df for t in trials], keys=keys)

    # free up some memory.
    [t.clear() for t in trials]

    df = svt(df, tol=tol, window=window)

    for t in trials:
        t.df = df.loc[(t.block.key, t.key), :].copy()
        lowpass(t.df, freq)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
