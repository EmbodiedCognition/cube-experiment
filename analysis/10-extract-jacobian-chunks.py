import climate
import joblib
import lmj.cubes
import logging
import numpy as np
import os


def extract(trial, output, frames):
    dirname = os.path.join(output, trial.subject.key)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    def save(arr, frames, targets, key):
        out = os.path.join(dirname, '{}_{}_{}_{}_{}.npy'.format(
            trial.block.key, trial.key, frames, targets, key))
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

    _, fjac, ijac = trial.jacobian(frames, inverse=True)

    # write out column names for three data types.
    with open(os.path.join(dirname, 'body-columns.txt'), 'w') as h:
        for c in body.columns:
            h.write(c + '\n')
    with open(os.path.join(dirname, 'goal-columns.txt'), 'w') as h:
        for c in goal.columns:
            h.write(c + '\n')
    with open(os.path.join(dirname, 'fjac-columns.txt'), 'w') as h:
        for c in fjac.columns:
            h.write(c + '\n')
    with open(os.path.join(dirname, 'ijac-columns.txt'), 'w') as h:
        for c in ijac.columns:
            h.write(c + '\n')

    start = frames
    starts = list(trial.df.target.diff(1).nonzero()[0])
    sources = ['0123456789ab'[int(i)] for i in trial.df.source.unique()]
    targets = ['0123456789ab'[int(i)] for i in trial.df.target.unique()]
    for i, end in enumerate(starts[1:] + [len(body)]):
        frm = '{:04d},{:04d}'.format(start, end)
        tgt = '{},{}'.format(sources[i], targets[i])
        if not (np.isnan(body.iloc[start:end].values).any() or
                np.isnan(goal.iloc[start:end].values).any() or
                np.isnan(fjac.iloc[start:end].values).any() or
                np.isnan(ijac.iloc[start:end].values).any()):
            save(body.iloc[start:end], frm, tgt, 'body')
            save(goal.iloc[start:end], frm, tgt, 'goal')
            save(fjac.iloc[start:end], frm, tgt, 'fjac')
            save(ijac.iloc[start:end], frm, tgt, 'ijac')
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
