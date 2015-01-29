#!/usr/bin/env python

import climate
import itertools
import joblib
import numpy as np
import os
import pandas as pd

import database

logging = climate.get_logger('jacobian')

GOAL_MARKERS = (
    'marker01-r-head-front',
    'marker06-r-collar',
    'marker08-r-elbow',
    'marker13-r-fing-index',
    'marker25-l-fing-index',
    'marker36-r-hip',
    'marker37-r-knee',
    'marker41-r-mt-outer',
)

def compute(trial, target, goal_markers=GOAL_MARKERS):
    trial.load()
    #trial.distance_to_target.plot(lw=2, alpha=0.7)
    trial.mask_fiddly_target_frames()
    #trial.distance_to_target.plot(lw=3, alpha=0.7)
    #plt.show()

    body = database.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body.add_velocities()

    goal = database.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal.add_velocities()

    for body_marker, body_channel in itertools.product(body.marker_columns, 'xyz'):
        bn = 'b{}{}'.format(body_marker[6:8], body_channel)
        bs = body.df['{}-v{}'.format(body_marker, body_channel)].copy()
        bs[bs == 0] = float('nan')
        for goal_marker, goal_channel in itertools.product(goal_markers, 'xyz'):
            gn = 'g{}{}'.format(goal_marker[6:8], goal_channel)
            gs = goal.df['{}-v{}'.format(goal_marker, goal_channel)].copy()
            gs[gs == 0] = float('nan')
            trial.df['jac-fwd-{}/{}'.format(gn, bn)] = gs / bs
            trial.df['jac-inv-{}/{}'.format(bn, gn)] = bs / gs

    trial.save(os.path.join(target, trial.parent.parent.basename, trial.parent.basename, trial.basename))


def main(root, target, pattern='*'):
    trials = database.Experiment(root).trials_matching(pattern)
    proc = joblib.delayed(compute)
    joblib.Parallel(-2)(proc(t, target) for t in trials)


if __name__ == '__main__':
    climate.call(main)
