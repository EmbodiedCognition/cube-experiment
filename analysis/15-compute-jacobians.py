#!/usr/bin/env python

import climate
import joblib
import lmj.cubes
import numpy as np
import os

logging = climate.get_logger('15-compute-jac')


def compute(trial, output, frames):
    trial.load()

    # encode body-relative data.
    body = lmj.cubes.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    stats = body.make_body_relative()
    #body.add_velocities()
    body_delta = body.diff(frames)

    # encode goal-relative data.
    goal = lmj.cubes.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    #goal.add_velocities()
    goal_delta = goal.diff(frames)

    out = lmj.cubes.Trial(trial.parent, trial.basename)
    out.df = trial.df.copy()
    for bc in body.marker_position_columns:
        for gc in goal.marker_position_columns:
            out.df['jac-fwd-{}/{}'] = goal_delta[gc] / body_delta[bc]
            out.df['jac-inv-{}/{}'] = body_delta[bc] / goal_delta[gc]

    out.save(trial.root.replace(trial.experiment.root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern='process trials matching this pattern',
)
def main(root, output, pattern='*'):
    work = joblib.delayed(compute)
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    #joblib.Parallel(-1)(work(t, output) for t in trials)
    compute(list(trials)[0], output)


if __name__ == '__main__':
    climate.call(main)
