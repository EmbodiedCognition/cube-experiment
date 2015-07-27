import climate
import joblib
import lmj.cubes
import logging
import numpy as np
import os


def extract(trial, output, frames):
    dirname = os.path.join(output, trial.subject.key)
    pattern = '{}-{}-{{}}.npy'.format(trial.block.key, trial.key)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    def save(key, arr):
        out = os.path.join(dirname, pattern.format(key))
        logging.info('%s: %s', out, arr.shape)
        np.save(out, arr.values)

    trial.load()
    for m in trial.marker_channel_columns:
        trial.df[m] = trial.df[m].interpolate()

    body = lmj.cubes.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body.add_velocities()
    body = body.df[body.marker_channel_columns]

    goal = lmj.cubes.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal.add_velocities()
    goal = goal.df[goal.marker_channel_columns]

    _, jac = trial.jacobian(frames)

    start = frames
    starts = list(trial.df.target.diff(1).nonzero()[0])
    for i, end in enumerate(starts[1:] + [len(body)]):
        save('body-{:02d}'.format(i), body.iloc[start:end])
        save('goal-{:02d}'.format(i), goal.iloc[start:end])
        save('jac-{:02d}'.format(i), jac.iloc[start:end])
        start = end + frames


@climate.annotate(
    root='load data from this root directory',
    output='save chunks to this directory',
    frames=('compute jacobian over this many frames', 'option', None, int),
)
def main(root, output, frames=10):
    trials = lmj.cubes.Experiment(root).trials_matching('*')
    work = joblib.delayed(extract)
    joblib.Parallel(-1)(work(t, output, frames) for t in trials)


if __name__ == '__main__':
    climate.call(main)
