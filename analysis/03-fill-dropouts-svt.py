#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.cubes.fill
import numpy as np
import pandas as pd

logging = climate.get_logger('fill')

def svt(dfs, threshold, window):
    '''Complete missing marker data using singular value thresholding.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Frames of source data. The frames will be stacked into a single large
        frame to use during SVT. This stacked frame will then be split and
        returned.
    threshold : float
        Threshold for singular values. If none, use a value computed from the
        spectrum of singular values.
    window : int
        Model windows of this many consecutive frames.
    '''
    df = lmj.cubes.fill.stack(dfs, window)
    centers = lmj.cubes.fill.center(df)
    pos, vis, data_norm = lmj.cubes.fill.window(df, window)

    # if the threshold is none, set it using the falloff point in the spectrum.
    if threshold is None:
        s = pd.Series(np.linalg.svd(pos, compute_uv=False))
        threshold = s[s.diff().diff().shift(-1).argmax()]
        logging.info('using threshold %.2f', threshold)

    i = 0
    dist = 1
    x = y = np.zeros_like(pos)
    while dist >= lmj.cubes.fill.PHASESPACE_TOLERANCE:
        i += 1
        err = vis * (pos - x)
        y += err
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u * s, v)
        dist = abs(err[vis]).mean()
        logging.info('%d: error %f (%d); mean %f; %s',
                     i, (err * err).sum() / data_norm, len(s.nonzero()[0]), dist,
                     np.percentile(abs(err[vis]), [50, 90, 95, 99]).round(4))

    lmj.cubes.fill.update(df, x, window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(svt, args, args.svt_threshold, args.window)


if __name__ == '__main__':
    climate.call(main)
