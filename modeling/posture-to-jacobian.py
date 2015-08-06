import climate
import fnmatch
import numpy as np
import os
import pandas as pd
import theanets


def matching(root, pattern):
    for root, dirs, files in os.walk(root):
        for fn in fnmatch.filter(files, pattern):
            yield os.path.join(root, fn)


def load_markers(fn):
    df = pd.read_csv(fn, index_col='time').dropna()
    cols = [c for c in df.columns if c.startswith('marker') and c[-1] in 'xyz']
    return df[cols].astype('f')


def load_jacobian(fn):
    df = pd.read_csv(fn, index_col='time').dropna()
    cols = [c for c in df.columns if c.startswith('jac')]
    return df[cols].astype('f')


def main(root):
    bodys = [load_markers(f) for f in sorted(matching(root, '*_body.csv.gz'))]
    goals = [load_markers(f) for f in sorted(matching(root, '*_goal.csv.gz'))]
    jacs = [load_jacobian(f) for f in sorted(matching(root, '*_jac.csv.gz'))]

    batch = 256

    nbody = bodys[0].shape[1]
    ngoal = goals[0].shape[1]
    njac = jacs[0].shape[1]

    net = theanets.Regressor([
        nbody + ngoal,
        1000,
        100,
        njac,
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='mean'),
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='covar'),
    ], )  # loss='gll')

    def paired():
        s = np.random.randint(len(bodys))
        body = bodys[s]
        goal = goals[s]
        jac = jacs[s]
        idx = body.index & goal.index & jac.index
        inputs = np.zeros((batch, nbody + ngoal), 'f')
        targets = np.zeros((batch, njac), 'f')
        for b in range(batch):
            o = np.random.choice(idx)
            inputs[b, :nbody] = body.loc[o, :]
            inputs[b, nbody:] = goal.loc[o, :]
            targets[b] = jac.loc[o, :]
        return [inputs, targets]

    net.train(
        paired,
        momentum=0.9,
        #input_noise=0.5,
        max_gradient_norm=10,
        monitors={
            'hid1:out': (0.0001, 0.001, 0.01, 0.1, 1),
        },
    )


if __name__ == '__main__':
    climate.call(main)
