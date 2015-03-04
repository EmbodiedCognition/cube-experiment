#!/usr/bin/env python

import climate
import lmj.cubes
import lmj.plot
import numpy as np
import pandas as pd


def main(root, pattern='*'):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        ds = pd.Series(t.df.index, index=t.df.index)
        dt = ds.shift(-1) - ds.shift(1)
        vbins = np.linspace(0, 12, 127)
        abins = np.linspace(0, 200, 127)
        vax = lmj.plot.create_axes(211, spines=True)
        aax = lmj.plot.create_axes(212, spines=True)
        for m in t.marker_columns:
            x = t.trajectory(m)
            v = x.diff(2).shift(-1).div(dt, axis='index')
            a = v.diff(2).shift(-1).div(dt, axis='index')
            vax.hist(np.sqrt((v ** 2).sum(axis=1)).dropna().values, bins=vbins, alpha=0.3, lw=0, log=True)
            aax.hist(np.sqrt((a ** 2).sum(axis=1)).dropna().values, bins=abins, alpha=0.3, lw=0, log=True)
        vax.set_xlim(0, 12)
        aax.set_xlim(0, 200)
        lmj.plot.show()
        break


if __name__ == '__main__':
    climate.call(main)
