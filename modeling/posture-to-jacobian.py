import climate
import lmj.cubes
import lmj.plot
import numpy as np
import pandas as pd
import theanets

logging = climate.get_logger('posture->jac')

BATCH = 256
THRESHOLD = 100


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

    '''
    with lmj.plot.axes() as ax:
        ax.hist(np.concatenate([j.values.ravel() for j in jacs]),
                bins=np.linspace(-THRESHOLD, THRESHOLD, 127), lw=0)
    '''

    net = theanets.Regressor([
        nbody,
        dict(size=1000, activation='sigmoid'),
        dict(size=1000, activation='sigmoid'),
        dict(size=1000, activation='sigmoid'),
        dict(size=1000, activation='sigmoid'),
        dict(size=1000, activation='sigmoid'),
        njac,
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='mean'),
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='covar'),
    ], loss='mse')  # 'gll')

    inputs = []
    outputs = []
    for s in range(len(bodys)):
        body = bodys[s]
        goal = goals[s]
        jac = jacs[s]
        idx = body.index & goal.index & jac.index
        inputs.append(body.loc[idx, :].values)
        outputs.append(np.clip(jac.loc[idx, :].values, -THRESHOLD, THRESHOLD))

    net.train(
        [np.vstack(inputs), np.vstack(outputs)],
        algo='layerwise',
        momentum=0.9,
        learning_rate=0.0001,
        patience=1,
        min_improvement=0.01,
        #max_gradient_norm=1,
        #input_noise=0.001,
        #hidden_l1=0.001,
        #hidden_dropout=0.1,
        monitors={
            #'*:out': (0.1, 0.5, 0.9),
        })

    net.save('/tmp/posture-jacobian.pkl.gz')


if __name__ == '__main__':
    climate.call(main)
