import climate
import joblib
import numpy as np


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

    _, jac = trial.jacobian(frames)

    start = frames
    for i, end in enumerate(trial.df.target.diff(1).nonzero()[0][1:]):
        save('body-{:02d}'.format(i), body.iloc[start:end])
        save('jac-{:02d}'.format(i), jac.iloc[start:end])
        start = end + frames


def main(root, output, frames=10):
    trials = lmj.cubes.Experiment(root).trials_matching('*')
    work = joblib.delayed(extract)
    joblib.Parallel(-1)(work(t, output, frames) for t in trials)


if __name__ == '__main__':
    climate.call(main)
