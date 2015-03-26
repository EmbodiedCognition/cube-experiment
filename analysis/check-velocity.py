#!/usr/bin/env python

import climate
import joblib
import lmj.cubes
import numpy as np


def _check(t):
    t.load()
    t.add_velocities(smooth=0)
    vel = abs(t.df[t.marker_velocity_columns].values).flatten()
    vel = vel[np.isfinite(vel)]
    pct = np.percentile(vel, [1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99])
    print(t.subject.key, t.block.key, t.key, *pct)


def main(root):
    trials = lmj.cubes.Experiment(root).trials_matching('*')
    check = joblib.delayed(_check)
    joblib.Parallel(-1)(check(t) for t in trials)


if __name__ == '__main__':
    climate.call(main)
