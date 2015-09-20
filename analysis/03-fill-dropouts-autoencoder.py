#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.cubes.fill
import numpy as np
import theanets

logging = climate.get_logger('fill')


def fill(dfs, rank, window):
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
    window : int
        Model windows of this many consecutive frames.
    '''
    df = lmj.cubes.fill.stack(dfs, window)
    centers = lmj.cubes.fill.center(df)
    pos, vis, _ = lmj.cubes.fill.window(df, window)

    d = pos.shape[1]
    net = theanets.Autoencoder((d, (int(rank), 'sigmoid'), d), weighted=True)
    for tm, _ in net.itertrain([pos.astype('f'), vis.astype('f')],
                               batch_size=128,
                               momentum=0.5):
        if tm['loss'] < lmj.cubes.fill.PHASESPACE_TOLERANCE:
            break

    batches = (net.predict(pos[o:o+64].astype('f')) for o in range(0, len(pos), 64))
    lmj.cubes.fill.update(df, np.concatenate(list(batches), axis=0), window)
    lmj.cubes.fill.restore(df, centers)
    lmj.cubes.fill.unstack(df, dfs)


def main(args):
    lmj.cubes.fill.main(fill, args, args.autoencoder_rank, args.window)


if __name__ == '__main__':
    climate.call(main)
