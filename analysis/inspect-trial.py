#!/usr/bin/env python

import climate
import IPython
import lmj.cubes
import lmj.plot
import numpy as np
import pandas as pd


def hist(trial, cols, bins=None):
    if bins is None:
        bins = np.linspace(-1000, 1000, 127)
    values = trial.df[cols].values.flatten()
    weights = np.ones_like(values) / len(values)
    lmj.plot.hist(values, bins=bins, lw=2, log=True, weights=weights, histtype='step')


def main(root, pattern):
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        IPython.embed()


if __name__ == '__main__':
    climate.call(main)
