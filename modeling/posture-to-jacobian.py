import climate
import glob
import lmj.cubes
import numpy as np
import os
import theanets


def main(root):
    files = glob.glob(os.path.join(root, '*_body.npy'))

    batch = 256
    nbody = np.load(files[0]).shape[1]
    njacobian = np.load(files[0].replace('_body', '_fjac_400')).shape[1]

    net = theanets.Regressor([
        nbody + nbody,
        300,
        300,
        dict(size=njacobian, activation='linear', inputs={'hid2:out': 300}, name='mean'),
        dict(size=njacobian, activation='relu', inputs={'hid2:out': 300}, name='covar'),
    ], loss='gll')

    def paired():
        s = np.random.randint(len(files))
        body = np.load(files[s])
        goal = np.load(files[s].replace('_body', '_goal'))
        fjac = np.load(files[s].replace('_body', '_fjac_400'))
        inputs = np.zeros((batch, nbody + nbody), 'f')
        targets = np.zeros((batch, njacobian), 'f')
        for b in range(batch):
            o = np.random.randint(len(body))
            inputs[b, :nbody] = body[o]
            inputs[b, nbody:] = goal[o]
            targets[b] = fjac[o]
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
