#!/usr/bin/env python

import climate
import glob
import os
import pagoda
import pagoda.viewer

def full(name):
    return os.path.join(os.path.dirname(__file__), name)

def main(root, subject, block='block00', trial='trial00'):
    w = pagoda.cooper.World(dt=1. / 100)
    w.erp = 0.3
    w.cfm = 1e-6
    w.load_skeleton(full('skeleton-{}.txt'.format(subject)))
    w.load_markers(
        glob.glob('{}/*{}/*{}/*{}*'.format(root, subject, block, trial))[0],
        full('markers-{}.txt'.format(subject)))
    pagoda.viewer.Viewer(w, paused=True).run()


if __name__ == '__main__':
    climate.call(main)
