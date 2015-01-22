#!/usr/bin/env python

import climate
import itertools
import joblib
import numpy as np
import os
import pandas as pd

import database

logging = climate.get_logger('jacobian')


def compute(trial, target, goal_markers=('marker13-r-fing-index', 'marker32-t3', 'marker35-r-ilium')):
    trial.load()
    trial.drop_empty_markers()
    #trial.drop_fiddly_target_frames()

    body = database.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body.add_velocities()

    goal = database.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    tgt = goal.target_trajectory
    goal.recenter(tgt.x, tgt.y, tgt.z)
    goal.add_velocities()

    for body_marker, body_channel in itertools.product(body.marker_columns, 'xyz'):
        bn = 'b{}{}'.format(body_marker[6:8], body_channel)
        bs = body.df['{}-v{}'.format(body_marker, body_channel)]
        bs[bs == 0] = 1e-8
        for goal_marker, goal_channel in itertools.product(goal_markers, 'xyz'):
            gn = 'g{}{}'.format(goal_marker[6:8], goal_channel)
            gs = goal.df['{}-v{}'.format(goal_marker, goal_channel)]
            gs[gs == 0] = 1e-8
            trial.df['jac-fwd-{}/{}'.format(gn, bn)] = gs / bs
            trial.df['jac-inv-{}/{}'.format(bn, gn)] = bs / gs

    r = os.path.join(target, trial.parent.parent.key, trial.parent.key)
    if not os.path.exists(r):
        os.makedirs(r)

    trial.df.to_pickle(os.path.join(r, '{}.pkl'.format(trial.key)))
    trial.df.to_csv(os.path.join(r, '{}.csv'.format(trial.key)))
    logging.info('%s %s %s: saved jacobian %s',
                 trial.parent.parent.key,
                 trial.parent.key,
                 trial.key,
                 trial.df.shape)


def main(root, target, pattern='*'):
    trials = database.Experiment(root).trials_matching(pattern, load=False)
    proc = joblib.delayed(compute)
    joblib.Parallel(-1)(proc(t, target) for t in trials)


if __name__ == '__main__':
    climate.call(main)
