#!/usr/bin/env python

from __future__ import division

import climate
import joblib
import lmj.cubes
import pandas as pd
import scipy.signal

logging = climate.get_logger('lowpass')

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
            df.loc[:, c] = scipy.signal.filtfilt(b, a, df[c].interpolate().ffill().bfill())


def work(t, root, output, freq):
    t.load()
    count = t.df.count().sum()
    lowpass(t.df, freq)
    logging.info('%s -> %s', count, t.df.count().sum())
    t.save(t.root.replace(root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    freq=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, pattern='*', freq=None):
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    run = joblib.delayed(work)
    args = root, output, freq
    joblib.Parallel(-1)(run(t, *args) for t in trials)


if __name__ == '__main__':
    climate.call(main)
