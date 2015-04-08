#!/usr/bin/env python

from __future__ import division

import climate
import lmj.cubes
import lmj.cubes.fill
import numpy as np
import theanets
import theanets.flags

logging = climate.get_logger('fill')

def fill(dfs, rank, tol, window):
    '''Complete missing marker data using a nonlinear autoencoder model.

    This method alters the given `dfs` in-place.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Frames of source data. The frames will be stacked into a single large
        frame to use during encoding. This stacked frame will then be split and
        returned.
    rank : int
        Encode the data using a nonlinear matrix decomposition of this rank.
    tol : float
        Quit optimizing the autoencoder once the loss drops below this
        threshold.
    window : int
        Model windows of this many consecutive frames.
    '''
    df = lmj.cubes.fill.stack(dfs, window)
    centers = lmj.cubes.fill.center(df)
    pos, vis, _ = lmj.cubes.fill.window(df, window)

    exp = theanets.Experiment(
        theanets.Autoencoder,
        layers=(pos.shape[1], rank, pos.shape[1]),
        weighted=True)
    exp.train([pos, vis], learning_rate=0.1, patience=1)
    for tm, _ in exp.itertrain([pos, vis]):
        if tm['loss'] < tol:
            break

    batches = (exp.network.predict(pos[o:o+64]) for o in range(0, len(pos), 64))
    lmj.cubes.fill.update(df, np.concatenate(list(batches), axis=0), window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(
        args, lambda ts: fill(
            [t.df for t in ts], args.rank, args.tol, args.window))


if __name__ == '__main__':
    climate.call(main)
