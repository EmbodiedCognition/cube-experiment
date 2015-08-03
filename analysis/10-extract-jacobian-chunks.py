import climate
import gzip
import io
import joblib
import lmj.cubes
import logging
import numpy as np
import os


TARGETS = '0123456789ab'


def extract(trial, root, output, frames):
    fn = trial.root.replace(root, output).replace('.csv.gz', '')

    def save(df, targets, key):
        if not os.path.isdir(os.path.dirname(fn)):
            os.makedirs(os.path.dirname(fn))
        out = fn + '_{}_{}.csv.gz'.format(targets, key)
        s = io.StringIO()
        df.to_csv(s, index_label='time')
        with gzip.open(out, 'w') as handle:
            handle.write(s.getvalue().encode('utf-8'))
        logging.info('%s: %s', out, df.shape)

    trial.load()

    body = lmj.cubes.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body.add_velocities()

    goal = lmj.cubes.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal.add_velocities()

    _, jac = trial.jacobian(frames)

    for t, target in enumerate(TARGETS):
        mask = trial.df.target == t
        if np.sum(mask) > 0:
            sources = trial.df[mask].source.unique()
            assert len(sources) == 1
            tgt = '{}{}'.format(TARGETS[int(sources[0])], target)
            save(body.df[mask], tgt, 'body')
            save(goal.df[mask], tgt, 'goal')
            save(jac[mask], tgt, 'jac')


@climate.annotate(
    root='load data from this root directory',
    output='save chunks to this directory',
    frames=('compute jacobian over this many frames', 'option', None, int),
)
def main(root, output, frames=10):
    trials = lmj.cubes.Experiment(root).trials_matching('*')
    work = joblib.delayed(extract)
    joblib.Parallel(-1)(work(t, root, output, frames) for t in trials)


if __name__ == '__main__':
    climate.call(main)
