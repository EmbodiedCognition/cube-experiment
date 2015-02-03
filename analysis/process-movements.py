#!/usr/bin/env python

import climate
import itertools
import joblib
import lmj.pca
import os
import pandas as pd
import random

import database

logging = climate.get_logger('compress')

MARKERS = [
    'marker00-r-head-back',
    'marker01-r-head-front',
    'marker02-l-head-front',
    'marker03-l-head-back',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
    'marker11-r-fing-ring',
    'marker12-r-fing-middle',
    'marker13-r-fing-index',
    'marker14-r-mc-outer',
    'marker15-r-mc-inner',
    'marker16-r-thumb-base',
    'marker17-r-thumb-tip',
    'marker18-l-collar',
    'marker19-l-shoulder',
    'marker20-l-elbow',
    'marker21-l-wrist',
    'marker22-l-fing-pinky',
    'marker23-l-fing-ring',
    'marker24-l-fing-middle',
    'marker25-l-fing-index',
    'marker26-l-mc-outer',
    'marker27-l-mc-inner',
    'marker28-l-thumb-base',
    'marker29-l-thumb-tip',
    'marker30-abdomen',
    'marker31-sternum',
    'marker32-t3',
    'marker33-t9',
    'marker34-l-ilium',
    'marker35-r-ilium',
    'marker36-r-hip',
    'marker37-r-knee',
    'marker38-r-shin',
    'marker39-r-ankle',
    'marker40-r-heel',
    'marker41-r-mt-outer',
    'marker42-r-mt-inner',
    'marker43-l-hip',
    'marker44-l-knee',
    'marker45-l-shin',
    'marker46-l-ankle',
    'marker47-l-heel',
    'marker48-l-mt-outer',
    'marker49-l-mt-inner',
]

COLUMNS = ['{}-{}'.format(m, c) for m in MARKERS for c in 'xyz']


def compress(trial, output, variance=0.995):
    trial.load()
    trial.mask_fiddly_target_frames()
    #trial.df.dropna(thresh=len(COLUMNS), inplace=True)

    init = [c for c in trial.columns if c[:6] in ('source', 'target')]
    out = pd.DataFrame(trial.df[init], index=trial.df.index)

    def p(w):
        return os.path.join(output, 'pca-{}-relative.npz'.format(w))

    # encode body-relative data.
    body = database.Trial(trial.parent, trial.basename)
    body.df = trial.df.copy()
    body.make_body_relative()
    body_pcs = 0

    pca = lmj.pca.PCA(filename=p('body'))
    for i, v in enumerate(pca.encode(body.df[COLUMNS].values, retain=variance).T):
        out['body-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
        body_pcs = i

    # encode goal-relative data.
    goal = database.Trial(trial.parent, trial.basename)
    goal.df = trial.df.copy()
    goal.make_target_relative()
    goal_pcs = 0

    pca = lmj.pca.PCA(filename=p('goal'))
    for i, v in enumerate(pca.encode(goal.df[COLUMNS].values, retain=variance).T):
        out['goal-pc{:02d}'.format(i)] = pd.Series(v, index=trial.df.index)
        goal_pcs = i

    # add columns for the jacobian.
    for bpc in range(body_pcs):
        db = out['body-pc{:02d}'.format(bpc)].diff()
        db[db == 0] = float('nan')
        for gpc in range(goal_pcs):
            dg = out['goal-pc{:02d}'.format(gpc)].diff()
            dg[dg == 0] = float('nan')
            out['jac-fwd-{:02d}/{:02d}'.format(gpc, bpc)] = dg / db
            out['jac-inv-{:02d}/{:02d}'.format(bpc, gpc)] = db / dg

    trial.df = out[sorted(out.columns)]
    trial.save(trial.root.replace(trial.experiment.root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save encoded data to this directory tree',
    pattern='process trials matching this pattern',
    variance=('retain this fraction of variance', 'option', None, float),
)
def main(root, output, pattern='*', variance=0.99):
    trials = list(database.Experiment(root).trials_matching(pattern))
    keys = [(t.block.key, t.key) for t in trials]
    probes = (variance, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)

    # choose N trials per subject to compute the principal components.
    N = 3
    pca_trials = []
    for s, ts in itertools.groupby(trials, key=lambda t: t.subject.key):
        ts = list(ts)
        idx = list(range(len(ts)))
        random.shuffle(idx)
        for i in idx[:N]:
            pca_trials.append(ts[i])
            ts[i].load()

    body = database.Movement(pd.concat([t.df for t in pca_trials]))
    body.make_body_relative()

    pca = lmj.pca.PCA()
    pca.fit(body.df[COLUMNS])
    for v in probes:
        print('{:.1f}%: {} body components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-body-relative.npz'))

    goal = database.Movement(pd.concat([t.df for t in pca_trials]))
    goal.make_target_relative()

    pca = lmj.pca.PCA()
    pca.fit(goal.df[COLUMNS])
    for v in probes:
        print('{:.1f}%: {} goal components'.format(100 * v, pca.num_components(v)))
    pca.save(os.path.join(output, 'pca-goal-relative.npz'))

    joblib.Parallel(-1)(joblib.delayed(compress)(t, output, variance) for t in trials)


if __name__ == '__main__':
    climate.call(main)
