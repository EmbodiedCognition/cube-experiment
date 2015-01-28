#!/usr/bin/env python

import climate
import collections
import joblib
import numpy as np
import pandas as pd

import database


def marker_channel_columns(df, min_filled=0):
    return [c for c in df.columns
            if c.startswith('marker') and
            c[-1] in 'xyz' and
            df[c].count() > min_filled * len(df)]


def svt(df, threshold=500, max_rmse=0.002, consec_frames=3, log_every=0):
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
    consec_frames : int, optional
        Compute the SVT using trajectories of this many consecutive frames.
        Defaults to 3.
    log_every : int, optional
        Number of SVT iterations between logging output. Defaults to 0,
        which only logs output at the start and finish of the SVT process.
    '''
    cols = marker_channel_columns(df, min_filled=0.01)

    marker_data = df[cols]
    num_frames, num_markers = marker_data.shape
    num_entries = num_frames * num_markers
    filled_ratio = marker_data.count().sum() / max(1e-3, num_entries)

    vals = marker_data.values
    extra = num_frames % consec_frames
    if extra > 0:
        fill = np.zeros((consec_frames - extra, num_markers))
        fill.fill(float('nan'))
        vals = np.vstack([vals, fill])
    df = pd.DataFrame(vals.reshape((len(vals) // consec_frames, num_markers * consec_frames)))

    logging.info('SVT: filling %d x %d, reshaped as %d x %d',
                 num_frames, num_markers, df.shape[0], df.shape[1])
    logging.info('SVT: missing %d of %d values (%.1f%% filled)',
                 num_entries - marker_data.count().sum(),
                 num_entries,
                 100 * filled_ratio)

    def log():
        logging.info('SVT %d: weighted rmse %f using %d singular values',
                     i, rmse, len(s.nonzero()[0]))

    def noise():
        return (max_rmse / 2) * np.random.randn(*err.shape)

    s = None
    x = y = pd.DataFrame(np.zeros_like(df))
    rmse = max_rmse + 1
    i = 0
    while rmse > max_rmse:
        err = (df - x).fillna(0)
        y += (1.2 / filled_ratio) * (err + noise())
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = pd.DataFrame(np.dot(u, np.dot(np.diag(s), v)))
        rmse = np.sqrt((err * err).mean().mean())
        if log_every and i % log_every == 0: log()
        i += 1
    log()

    x = np.asarray(x).reshape((-1, num_markers))[:num_frames]
    filled = pd.DataFrame(x, index=marker_data.index, columns=marker_data.columns)
    for c in cols:
        df[c] = filled[c]


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
