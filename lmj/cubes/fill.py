import climate
import numpy as np
import pandas as pd

logging = climate.get_logger(__name__)

g = climate.add_group('dropout-filling options')
g.add_argument('--root', metavar='DIR',
               help='load data files from tree at DIR')
g.add_argument('--output', metavar='DIR',
               help='save smoothed data files to tree at DIR')
g.add_argument('--pattern', default='*', metavar='SHPAT',
               help='process only trials matching this pattern')
g.add_argument('--rank', type=int, metavar='K',
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


def assemble(dfs, window):
    '''Assemble multiple dfs into a single df with a multiindex.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Data frames for source data. The frames will be stacked into a single
        large frame to use while filling dropouts. This stacked frame can be
        split up using :meth:`disassemble`.
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

    num_frames, num_channels = data.shape
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
        dict(x=filled[[m + '-x' for m in CENTERS]].mean(axis=1),
             y=filled[[m + '-y' for m in CENTERS]].mean(axis=1),
             z=filled[[m + '-z' for m in CENTERS]].mean(axis=1)))
    for c in df.columns:
        df[c] -= center[c[-1]]
    return center


def restore(dfs, df, centers):
    '''Restore data in the given frame to the given center locations.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Original source data frames.
    df : pd.DataFrame
        Data frame containing stacked, centered mocap data.
    centers : pd.DataFrame
        Frame containing center locations for each time step in the recording.
    '''
    # shift data back out to the world.
    for c in df.columns:
        df.loc[:, c] += centers.loc[:, c[-1]]
    # unstack the stacked data frame.
    for i, d in enumerate(dfs):
        d[df.columns] = df.loc[(i, ), :]


def window(df, window, interpolate=False):
    '''
    '''
    visible = (~df.isnull()).values
    if interpolate:
        # interpolate dropouts linearly.
        position = df.interpolate().ffill().bfill().values
    else:
        # just fill dropouts with zeros.
        position = df.fillna(0).values
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
        [position[i:num_frames-(window-1-i)] for i in range(window)], axis=1)
    vis = np.concatenate(
        [visible[i:num_frames-(window-1-i)] for i in range(window)], axis=1)
    data_norm = (vis * pos * pos).sum()
    logging.info('processing windowed data %s: norm %s', pos.shape, data_norm)
    assert np.isfinite(data_norm)
    return pos, vis, data_norm
