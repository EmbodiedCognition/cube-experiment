#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import numpy as np
import pandas as pd
import scipy.signal
import theanets
import theanets.flags

logging = climate.get_logger('fill')

g = climate.add_group('svt options')
g.add_argument('--root', metavar='DIR', help='load data files from tree at DIR')
g.add_argument('--output', metavar='DIR', help='save smoothed data files to tree at DIR')
g.add_argument('--pattern', default='*', help='process only trials matching this pattern')
g.add_argument('--rank', type=int, help='reconstruction rank')
g.add_argument('--tol', default=0.001, type=float, help='reconstruction error tolerance')
g.add_argument('--window', type=int, help='process windows of T frames')
g.add_argument('--freq', type=float, help='lowpass filter at N Hz')

def fill(dfs, rank, tol, window):
    '''Complete missing marker data using a nonlinear autoencoder model.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Data frames for source data. Each frame will be interpolated, and then
        the frames will be stacked into a single large frame to use during
        encoding. This stacked frame will then be split and returned.
    rank : int
        Encode the data using a nonlinear matrix decomposition of this rank.
    tol : float
        Quit optimizing the autoencoder once the loss drops below this
        threshold.
    window : int
        Model windows of this many consecutive frames.
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

    weights = (~data.isnull()).values.astype('f')
    filled = data.fillna(0).values.astype('f')

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
    # this stacked data matrix is the one we'll want to fill in the encoding
    # process below.
    pos = np.concatenate([filled[i:num_frames-(window-1-i)] for i in range(window)], axis=1)
    wgt = np.concatenate([weights[i:num_frames-(window-1-i)] for i in range(window)], axis=1)

    data_norm = (wgt * pos * pos).sum()
    logging.info('processing windowed data %s: norm %s', pos.shape, data_norm)
    assert np.isfinite(data_norm)

    exp = theanets.Experiment(
        theanets.Autoencoder,
        layers=(pos.shape[1], rank, pos.shape[1]),
        weighted=True)
    exp.enable_command_line()
    for tm, _ in exp.itertrain([pos, wgt]):
        if tm['loss'] < tol:
            break

    w = window - 1
    rows = pd.MultiIndex(levels=data.index.levels,
                         labels=[x[w:-w] for x in data.index.labels])
    batches = (exp.network.predict(pos[o:o+64]) for o in range(0, len(t), 64))
    parts = np.split(np.concatenate(list(batches), axis=0), window, axis=1)
    aligned = [p[w - j:num_frames - w - j] for j, p in enumerate(parts)]
    filled = pd.DataFrame(np.mean(aligned, axis=0), index=rows, columns=cols)
    data.update(data.loc[rows, cols].fillna(filled))

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


def main(args):
    for _, ts in lmj.cubes.Experiment(args.root).by_subject(args.pattern):
        ts = list(ts)
        for t in ts:
            t.load()
            t.mask_dropouts()
        try:
            fill([t.df for t in ts], args.rank, args.tol, args.window)
        except Exception as e:
            logging.exception('error filling dropouts!')
            continue
        for i, t in enumerate(ts):
            if args.freq:
                lowpass(t.df, args.freq)
            t.save(t.root.replace(args.root, args.output))


if __name__ == '__main__':
    climate.call(main)
