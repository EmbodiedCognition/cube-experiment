#!/usr/bin/env python

import climate
import joblib
import lmj.pca

import database


def jac(trial):
    trial.load()
    cols = [c for c in trial.df.columns if c.startswith('jac-fwd')]
    return trial.df[cols].values


def main(root, pattern='*'):
    trials = database.Experiment(root).trials_matching(pattern)
    proc = joblib.delayed(jac)
    jacobians = []
    for jacs in joblib.Parallel(-2)(proc(t) for t in trials):
        jacobians.extend(jacs)
    print(len(jacobians))


if __name__ == '__main__':
    climate.call(main)
