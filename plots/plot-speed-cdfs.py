import climate
import itertools
import lmj.plot
import numpy as np
import re

PATTERN = re.compile(r'(\S+) (\S+) (\S+) (\S+). (\d+) frames, masking (\d+) = (\d+) dropouts \+ (\d+) spd \((.*)\) \+ (\d+) acc \((.*)\)')


def extract(line):
    sub, blk, tri, mrk, n, m, d, s, ss, a, aa = PATTERN.search(line).groups()
    n, m, d, s, a = (int(x) for x in (n, m, d, s, a))
    speeds = {}
    for x in ss.split():
        c, t = x.split('<')
        speeds[float(t)] = int(c) / n
    accels = {}
    for x in aa.split():
        c, t = x.split('<')
        accels[float(t)] = int(c) / n
    return sub, blk, tri, mrk, m / n, d / n, s / n, a / n, speeds, accels


def main(log):
    spd_by_marker, spd_by_subject = {}, {}
    acc_by_marker, acc_by_subject = {}, {}
    with open(log) as h:
        for l in h:
            try:
                sub, blk, tri, mrk, m, d, s, a, spd, acc = extract(l)
                spd_by_marker.setdefault(mrk, []).append(spd)
                spd_by_subject.setdefault(sub, []).append(spd)
                acc_by_marker.setdefault(mrk, []).append(acc)
                acc_by_subject.setdefault(sub, []).append(acc)
            except AttributeError:
                continue

    ax = lmj.plot.create_axes(121, spines=True)
    for c, (l, ss) in zip(itertools.cycle(lmj.plot.COLOR11), spd_by_marker.items()):
        xs = np.array(sorted(ss[0].keys()))
        ys = np.array([sorted(s.values()) for s in ss])
        m = ys.mean(axis=0)
        e = ys.std(axis=0)
        ax.plot(xs, m, color=c, label=l, alpha=0.7)
        ax.fill_between(xs, m + e, m - e, color=c, lw=0, alpha=0.3)
    ax.set_title('By Marker')
    ax.set_ylim(0.8, 1)
    #ax.set_yscale('log')
    #ax.legend()

    ax = lmj.plot.create_axes(122, spines=True)
    for c, (l, ss) in zip(itertools.cycle(lmj.plot.COLOR11), spd_by_subject.items()):
        xs = np.array(sorted(ss[0].keys()))
        ys = np.array([sorted(s.values()) for s in ss])
        m = ys.mean(axis=0)
        e = ys.std(axis=0)
        ax.plot(xs, m, color=c, label=l, alpha=0.7)
        ax.fill_between(xs, m + e, m - e, color=c, lw=0, alpha=0.3)
    ax.set_title('By Subject')
    ax.set_ylim(0.8, 1)
    #ax.set_yscale('log')
    #ax.legend()
    lmj.plot.show()


if __name__ == '__main__':
    climate.call(main)
