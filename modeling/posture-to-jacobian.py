import climate
import lmj.cubes
import numpy as np
import pandas as pd
import theanets

logging = climate.get_logger('posture->jac')

BATCH = 256


def load_markers(fn):
    df = pd.read_csv(fn, index_col='time').dropna()
    cols = [c for c in df.columns if c.startswith('marker') and c[-1] in 'xyz']
    return df[cols].astype('f')


def load_jacobian(fn):
    df = pd.read_csv(fn, index_col='time').dropna()
    cols = [c for c in df.columns if c.startswith('pc')]
    return df[cols].astype('f')


def main(root):
    match = lmj.cubes.utils.matching

    bodys = [load_markers(f) for f in sorted(match(root, '*_body.csv.gz'))]
    nbody = bodys[0].shape[1]
    logging.info('loaded %d body-relative files', len(bodys))

    goals = [load_markers(f) for f in sorted(match(root, '*_goal.csv.gz'))]
    ngoal = goals[0].shape[1]
    logging.info('loaded %d goal-relative files', len(goals))

    jacs = [load_jacobian(f) for f in sorted(match(root, '*_jac_pca23.csv.gz'))]
    njac = jacs[0].shape[1]
    logging.info('loaded %d jacobian files', len(jacs))

    net = theanets.Regressor([
        nbody + ngoal,
        1000,
        njac,
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='mean'),
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='covar'),
    ], )  # loss='gll')

    def paired():
        inputs = np.zeros((BATCH, nbody + ngoal), 'f')
        targets = np.zeros((BATCH, njac), 'f')
        for b in range(BATCH):
            s = np.random.randint(len(bodys))
            body = bodys[s]
            goal = goals[s]
            jac = jacs[s]
            o = np.random.choice(body.index & goal.index & jac.index)
            inputs[b, :nbody] = body.loc[o, :]
            inputs[b, nbody:] = goal.loc[o, :]
            targets[b] = jac.loc[o, :]
        return [inputs, targets]

    net.train(
        paired,
        algo='nag',
        momentum=0.9,
        input_noise=0.001,
        max_gradient_norm=1,
        learning_rate=0.0001,
        monitors={
            'hid1:out': (0.0001, 0.001, 0.01, 0.1, 1),
        },
    )


if __name__ == '__main__':
    climate.call(main)
