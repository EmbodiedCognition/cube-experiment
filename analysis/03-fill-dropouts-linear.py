#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import lmj.cubes.fill
import numpy as np
import pandas as pd

logging = climate.get_logger('fill')

def fill(dfs, window):
    '''Complete missing marker data using linear interpolation.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Frames of source data. The frames will be stacked into a single large
        frame to use during SVT. This stacked frame will then be split and
        returned.
    window : int
        Model windows of this many consecutive frames.
    '''
    df = lmj.cubes.fill.stack(dfs, window)
    centers = lmj.cubes.fill.center(df)
    pos, _, _ = lmj.cubes.fill.window(df, window, interpolate=True)
    lmj.cubes.fill.update(df, pos, window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(args, lambda ts: fill([t.df for t in ts], args.window))


if __name__ == '__main__':
    climate.call(main)
