#!/usr/bin/env python

import climate
import joblib
import numpy as np
import os
import pandas as pd

import database

logging = climate.get_logger('jacobian')


def compute(trial, target, goal_markers=('marker13-r-fing-index', 'marker32-t3')):
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

    fwd = pd.DataFrame(goal.df[['source', 'target']])
    inv = pd.DataFrame(goal.df[['source', 'target']])
    for body_marker in body.marker_columns:
        for body_channel in 'xyz':
            bc = '{}-v{}'.format(body_marker, body_channel)
            bn = 'body-{}-{}'.format(body_marker, body_channel)

            for goal_marker in goal_markers:
                for goal_channel in 'xyz':
                    gc = '{}-v{}'.format(goal_marker, goal_channel)
                    gn = 'goal-{}-{}'.format(goal_marker, goal_channel)
                    fwd['{}/{}'.format(gn, bn)] = goal.df[gc] / body.df[bc]
                    inv['{}/{}'.format(bn, gn)] = body.df[bc] / goal.df[gc]

    r = os.path.join(target, trial.parent.parent.key, trial.parent.key)
    if not os.path.exists(r):
        os.makedirs(r)

    fwd.to_pickle(os.path.join(r, '{}-forward.pkl'.format(trial.key)))
    inv.to_pickle(os.path.join(r, '{}-inverse.pkl'.format(trial.key)))
    logging.info('%s %s %s: saved jacobian %s',
                 trial.parent.parent.key,
                 trial.parent.key,
                 trial.key,
                 fwd.shape)


def main(root, target, pattern='*'):
    trials = database.Experiment(root).trials_matching(pattern, load=False)
    proc = joblib.delayed(compute)
    joblib.Parallel(-1)(proc(t, target) for t in trials)


if __name__ == '__main__':
    climate.call(main)
