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

    net = theanets.recurrent.Regressor([
        nbody,
        (200, 'sigmoid', 'gru'),
        (200, 'sigmoid', 'gru'),
        (200, 'sigmoid', 'gru'),
        (200, 'sigmoid', 'gru'),
        (200, 'sigmoid', 'gru'),
        njac,
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='mean'),
        #dict(size=njac, activation='linear', inputs={'hid2:out': 300}, name='covar'),
    ], loss='mse')

    '''
    inputs = []
    outputs = []
    for s in range(len(bodys)):
        body = bodys[s]
        goal = goals[s]
        jac = jacs[s]
        idx = body.index & goal.index & jac.index
        b = body.loc[idx, :].values
        g = goal.loc[idx, :].values
        inputs.append(g)#np.hstack([b, g]))
        outputs.append(np.clip(jac.loc[idx, :].values, -THRESHOLD, THRESHOLD))
    '''

    B = 32

    def batches(T):
        inputs = np.zeros((B, T, nbody), 'f')
        outputs = np.zeros((B, T, njac), 'f')

        def batch():
            for b in range(B):
                idx = []
                while len(idx) <= T:
                    s = np.random.randint(len(bodys))
                    body = bodys[s]
                    goal = goals[s]
                    jac = jacs[s]
                    idx = body.index & goal.index & jac.index
                i = np.random.randint(len(idx) - T)
                inputs[b] = body.loc[idx, :].iloc[i:i+T, :].values
                outputs[b] = np.clip(jac.loc[idx, :].iloc[i:i+T, :].values,
                                     -THRESHOLD, THRESHOLD)
            return [inputs, outputs]

        return batch

    net.train(
        #[np.vstack(inputs), np.vstack(outputs)],
        batches(32),
        algo='layerwise',
        momentum=0.9,
        learning_rate=0.0001,
        patience=5,
        min_improvement=0.01,
        #max_gradient_norm=1,
        #input_noise=0.001,
        #hidden_l1=0.001,
        #hidden_dropout=0.1,
        monitors={
            #'*:out': (0.1, 0.5, 0.9),
        })

    net.save('/tmp/posture-jacobian-gru.pkl.gz')


if __name__ == '__main__':
    climate.call(main)
