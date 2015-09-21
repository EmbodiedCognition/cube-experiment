import climate
import lmj.cubes
import lmj.plot
import numpy as np

import utils


def load(root, pattern):
    for trial in lmj.cubes.Experiment(root).trials_matching(pattern):
        trial.load()
        trial.add_velocities(7)
        l = trial.trajectory(47, velocity=True)
        l['speed'] = np.sqrt((l[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
        lm = utils.local_minima(l.speed.values)
        r = trial.trajectory(40, velocity=True)
        r['speed'] = np.sqrt((r[['vx', 'vy', 'vz']] ** 2).sum(axis=1))
        rm = utils.local_minima(r.speed.values)
        yield list(lm), l.iloc[lm, :], list(rm), r.iloc[rm, :]


def main(root, pattern='*'):
    legend = True
    with lmj.plot.axes(aspect='equal') as ax:
        ax.scatter([0], [0], color='#111111', alpha=0.7, s=200, marker='s', lw=0)
        for lm, left, rm, right in load(root, pattern):
            x, z = [], []
            while lm and rm:
                if lm[0] < rm[0]:
                    x.append(left.iloc[lm[0], 0])
                    z.append(left.iloc[lm[0], 2])
                    lm.pop(0)
                else:
                    x.append(right.iloc[rm[0], 0])
                    z.append(right.iloc[rm[0], 2])
                    rm.pop(0)
            if lm:
                x.extend(left.iloc[lm, 0])
                z.extend(left.iloc[lm, 2])
            if rm:
                x.extend(right.iloc[rm, 0])
                z.extend(right.iloc[rm, 2])
            ax.plot(x, z, color='#111111', lw=1, alpha=0.5)
            ax.scatter(left.x, left.z, color='#ff7f0e', label=legend and 'Left Foot')
            ax.scatter(right.x, right.z, color='#1f77b4', label=legend and 'Right Foot')
            legend = False
        lmj.plot.legend(loc='upper left')


if __name__ == '__main__':
    climate.call(main)
