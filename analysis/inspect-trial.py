#!/usr/bin/env python

import climate
import IPython
import lmj.cubes
import lmj.plot
import numpy as np
import pandas as pd


def main(root, pattern):
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        IPython.embed()


if __name__ == '__main__':
    climate.call(main)
