#!/usr/bin/env python

import climate
import lmj.cubes


def main(root, pattern='*'):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        t.pickle(t.root.replace('.csv.gz', '.pkl'))


if __name__ == '__main__':
    climate.call(main)
