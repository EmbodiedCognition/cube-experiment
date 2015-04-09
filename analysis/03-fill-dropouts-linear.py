#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import lmj.cubes.fill
import lmj.pca
import numpy as np
import pandas as pd

logging = climate.get_logger('fill-linear')

def fill(dfs, rank, window):
    '''Complete missing marker data using linear interpolation.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Frames of source data. The frames will be stacked into a single large
        frame to use during SVT. This stacked frame will then be split and
        returned.
    rank : float
        Number of principal components (if >1) or fraction of variance (if in
        (0, 1)) to retain in the encoded data.
    window : int
        Model windows of this many consecutive frames.
    '''
    df = lmj.cubes.fill.stack(dfs, window)
    centers = lmj.cubes.fill.center(df)
    if rank is None:
        prediction, _, _ = lmj.cubes.fill.window(df, window, True)
    else:
        if not 0 < rank < 1:
            rank = int(rank)
        pca = lmj.pca.PCA()
        pos, _, _ = lmj.cubes.fill.window(df, window, None)
        pca.fit(pos)
        enc = pd.DataFrame(pca.encode(pos, retain=rank))
        lin = enc.interpolate().ffill().bfill().values
        prediction = pca.decode(lin, retain=rank)
    lmj.cubes.fill.update(df, prediction, window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(args, lambda ts: fill([t.df for t in ts], args.window))


if __name__ == '__main__':
    climate.call(main)
