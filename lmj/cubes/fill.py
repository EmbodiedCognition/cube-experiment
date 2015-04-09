import climate
import numpy as np
import pandas as pd

from .database import Experiment

logging = climate.get_logger(__name__)

g = climate.add_group('dropout-filling options')
g.add_argument('--root', metavar='DIR',
               help='load data files from tree at DIR')
g.add_argument('--output', metavar='DIR',
               help='save smoothed data files to tree at DIR')
g.add_argument('--pattern', default='*', metavar='SHPAT',
               help='process only trials matching this pattern')
g.add_argument('--rank', type=float, metavar='K',
               help='reconstruction rank')
g.add_argument('--threshold', type=float, metavar='S',
               help='truncate singular values at threshold S')
g.add_argument('--tol', type=float, metavar='E',
               help='reconstruction error tolerance')
g.add_argument('--window', type=int, metavar='T',
               help='process windows of T frames')

CENTERS = [
    'marker34-l-ilium',
    'marker35-r-ilium',
    'marker36-r-hip',
    'marker37-r-knee',
]


def stack(dfs, window):
    '''Assemble multiple dfs into a single df with a multiindex.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Data frames for source data. The frames will be stacked into a single
        large frame to use while filling dropouts. This stacked frame can be
        split up using :meth:`unstack`.
    window : int
        Window length for filling dropouts.

    Returns
    -------
    data : pd.DataFrame
        A stacked data frame.
    '''
    cols = [c for c in dfs[0].columns if c.startswith('marker') and c[-1] in 'xyz']
    pad = pd.DataFrame(float('nan'), index=list(range(window - 1)), columns=cols)
    chunks, keys = [pad], [-1]
    for i, df in enumerate(dfs):
        chunks.extend([df[cols], pad])
        keys.extend([i, -len(chunks)])

    df = pd.concat(chunks, axis=0, keys=keys)

    num_frames, num_channels = df.shape
    num_entries = num_frames * num_channels
    filled_ratio = df.count().sum() / num_entries
    logging.info('missing %d of (%d, %d) = %d values (%.1f%% filled)',
                 num_entries - df.count().sum(),
                 num_frames, num_channels, num_entries,
                 100 * filled_ratio)

    # if a column is completely missing, refuse to process the data.
    for c in df.columns:
        if df[c].count() == 0:
            raise ValueError('%s: no visible values!', c)

    return df


def center(df):
    '''Shift an entire data frame by the location of the CENTERS.

    The centers are basically the hip markers; this moves the data so that all
    markers are centered on the origin (as a group).

    Parameters
    ----------
    df : pd.DataFrame
        A data frame containing motion-capture marker locations.

    Returns
    -------
    center : pd.DataFrame
        A frame representing the center of the markers at each time step in the
        original data frame.
    '''
    center = pd.DataFrame(
        dict(x=df[[m + '-x' for m in CENTERS]].mean(axis=1),
             y=df[[m + '-y' for m in CENTERS]].mean(axis=1),
             z=df[[m + '-z' for m in CENTERS]].mean(axis=1)))
    for c in df.columns:
        df[c] -= center[c[-1]]
    return center


def restore(df, centers):
    '''Restore data in the given frame to the given center locations.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing stacked, centered mocap data.
    centers : pd.DataFrame
        Frame containing center locations for each time step in the recording.
    '''
    for c in df.columns:
        df.loc[:, c] += centers.loc[:, c[-1]]


def unstack(df, dfs):
    '''Unstack a stacked frame into multiple individual frames.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing stacked, centered mocap data.
    dfs : list of pd.DataFrame
        Individual target data frames.
    '''
    for i, d in enumerate(dfs):
        d[df.columns] = df.loc[(i, ), :]


def window(df, window, fillna=0):
    '''Create windowed arrays of marker position data.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing stacked marker position data.
    window : int
        Number of successive frames to include in each window.
    fillna : float, int, str, or None
        If an integer or float, fill dropouts with this value. If a string,
        interpolate missing data linearly. If None, do not fill dropouts.
    '''
    visible = (~df.isnull()).values
    position = df.values
    if isinstance(fillna, (float, int)):
        # just fill dropouts with a constant value.
        position = df.fillna(fillna).values
    if fillna is not None:
        # interpolate dropouts linearly.
        position = df.interpolate().ffill().bfill().values
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
    pos = np.concatenate(
        [position[i:len(df)-(window-1-i)] for i in range(window)], axis=1)
    vis = np.concatenate(
        [visible[i:len(df)-(window-1-i)] for i in range(window)], axis=1)
    data_norm = (vis * pos * pos).sum()
    logging.info('processing windowed data %s: norm %s', pos.shape, data_norm)
    assert np.isfinite(data_norm)
    return pos, vis, data_norm


def update(df, prediction, window, only_dropouts=True):
    '''Update a stacked data frame using predicted marker positions.

    Parameters
    ----------
    df : pd.DataFrame
        Stacked data frame to update with predicted marker locations.
    prediction : ndarray
        Array of predicted marker locations.
    window : int
        Window size. Windows will be unstacked and averaged before updating.
    only_dropouts : bool
        If True (default), only fill in values for dropouts in the original data
        frame. If False, replace all values in the original frame with
        predictions.
    '''
    # above, we created <window> duplicates of our data, each offset by 1 frame,
    # and stacked (along axis 1) into a big ol matrix. in effect, we have
    # <window> copies of each frame; here, we unpack these duplicates, remove
    # their offsets, take the average, and put them back in the linear frame.
    w = window - 1
    cols = df.columns
    rows = pd.MultiIndex(
        levels=df.index.levels, labels=[x[w:-w] for x in df.index.labels])
    parts = np.split(prediction, window, axis=1)
    mean = np.mean([p[w - j:len(p) - j] for j, p in enumerate(parts)], axis=0)
    if only_dropouts:
        df.update(df.loc[rows, cols].fillna(
            pd.DataFrame(mean, index=rows, columns=cols)))
    else:
        df.loc[rows, cols] = mean


def main(fill, args, *fill_args):
    for _, ts in Experiment(args.root).by_subject(args.pattern):
        ts = list(ts)
        for t in ts:
            t.load()
            t.mask_dropouts()
        try:
            fill([t.df for t in ts], *fill_args)
        except Exception as e:
            logging.exception('error filling dropouts!')
            continue
        [t.save(t.root.replace(args.root, args.output)) for t in ts]
