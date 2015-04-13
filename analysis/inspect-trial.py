#!/usr/bin/env python

import climate
import lmj.cubes
import IPython


def main(root, pattern):
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        IPython.embed()


if __name__ == '__main__':
    climate.call(main)
