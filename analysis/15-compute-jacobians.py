#!/usr/bin/env python

import climate
import joblib
import lmj.cubes

logging = climate.get_logger('15-compute-jac')


def compute(trial, output, frames):
    trial.load()

    # encode body-relative data.
    body = lmj.cubes.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    stats = body.make_body_relative()
    #body.add_velocities()
    body_delta = body.df.diff(frames)
    logging.info('computed body delta')

    # encode goal-relative data.
    goal = lmj.cubes.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    #goal.add_velocities()
    goal_delta = goal.df.diff(frames)
    logging.info('computed goal delta')

    out = lmj.cubes.Trial(trial.parent, trial.basename)
    out.df = trial.df.copy()
    for bc in body.marker_position_columns:
        bn = 'b{}{}'.format(bc[6:8], bc[-1])
        for gc in goal.marker_position_columns:
            gn = 'g{}{}'.format(gc[6:8], gc[-1])
            out.df['jac-fwd-{}/{}'.format(gn, bn)] = goal_delta[gc] / body_delta[bc]
            out.df['jac-inv-{}/{}'.format(bn, gn)] = body_delta[bc] / goal_delta[gc]
        logging.info('computed body columns %s', bn)

    out.save(trial.root.replace(trial.experiment.root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern='process trials matching this pattern',
    frames=('compute jacobian over this many frames', 'option', None, int),
)
def main(root, output, pattern='*', frames=20):
    work = joblib.delayed(compute)
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    joblib.Parallel(2)(work(t, output, frames) for t in trials)


if __name__ == '__main__':
    climate.call(main)
