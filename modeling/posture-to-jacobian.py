import climate
import lmj.cubes
import numpy as np
import theanets


def main(root, pattern, frames=10):

    net = theanets.Regressor([
        poss[0].shape[1],
        dict(name='latent', size=1000),
        300,
        jacs[0].shape[1],
    ])

    def paired():
        p = np.zeros((64, poss[0].shape[1]), 'f')
        j = np.zeros((64, jacs[0].shape[1]), 'f')
        for i in range(len(p)):
            s = np.random.randint(len(poss))
            ps, js = poss[s], jacs[s]
            o = 0
            while any(js[o].isnull()):
                o = np.random.randint(len(ps))
            p[i], j[i] = ps[o], js[o]
        return [p, j]

    net.train(
        paired,
        input_noise=1,
        hidden_l1={'latent:out': 1},
        monitors={'hid1:out': (0, 0.01, 0.1, 1)},
    )


if __name__ == '__main__':
    climate.call(main)
