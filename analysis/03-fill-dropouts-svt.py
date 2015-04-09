#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import lmj.cubes.fill
import numpy as np
import pandas as pd

logging = climate.get_logger('fill')

def svt(dfs, tol, threshold, window):
    '''Complete missing marker data using singular value thresholding.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Frames of source data. The frames will be stacked into a single large
        frame to use during SVT. This stacked frame will then be split and
        returned.
    tol : float
        Continue the reconstruction process until reconstructed data is within
        this relative tolerance threshold.
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
    err = 1e200
    x = y = np.zeros_like(pos)
    while tol * data_norm < err:
        e = vis * (pos - x)
        err = (e * e).sum()
        y += e
        u, s, v = np.linalg.svd(y, full_matrices=False)
        s = np.clip(s - threshold, 0, np.inf)
        x = np.dot(u * s, v)
        logging.info('%d: error %f (%d); mean %f; %s',
                     i, err / data_norm, len(s.nonzero()[0]), abs(e[vis]).mean(),
                     np.percentile(abs(e[vis]), [50, 90, 95, 99]).round(4))
        i += 1

    lmj.cubes.fill.update(df, x, window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(svt, args, args.tol, args.threshold, args.window)


if __name__ == '__main__':
    climate.call(main)
