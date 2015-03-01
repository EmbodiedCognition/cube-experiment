#!/usr/bin/env python

import climate
import joblib
import lmj.cubes
import numpy as np
import pandas as pd

logging = climate.get_logger('reindex')

# this is the set of markers that gets included in our output.
MARKERS = [
    'marker00-r-head-back',
    'marker01-r-head-front',
    'marker02-l-head-front',
    'marker03-l-head-back',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
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


def reindex(t, root, output, frame_rate, interpolate, max_speed, max_acc):
    t.load()

    # drop unusable marker data.
    for c in t.columns:
        if c.startswith('marker') and c[:-2] not in MARKERS:
            del t.df[c]

    # mask marker frames that are dropouts or are moving too fast.
    ds = pd.Series(t.df.index, index=t.df.index)
    dt = ds.shift(-1) - ds.shift(-1)
    for marker in t.marker_columns:
        cols = ['{}-{}'.format(marker, z) for z in 'xyzc']
        x, y, z, c = (self.df[c] for c in cols)
        drops = c.isnull() | (c < 0) | (c > 100)
        vx = x.diff(2).shift(-1) / dt
        vy = y.diff(2).shift(-1) / dt
        vz = z.diff(2).shift(-1) / dt
        speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) > max_speed
        ax = vx.diff(2).shift(-1) / dt
        ay = vy.diff(2).shift(-1) / dt
        az = vz.diff(2).shift(-1) / dt
        acc = np.sqrt(ax ** 2 + ay ** 2 + az ** 2) > max_acc
        mask = drops | speed | acc
        logging.info('%s %s %s %s: masking %d: %d dropouts, %d vel, %d acc',
                     t.subject.key, t.block.key, t.key, marker,
                     mask.sum(), drops.sum(), speed.sum(), acc.sum())

    # reindex to regularly-spaced temporal index values.
    posts = np.arange(0, t.df.index[-1], 1. / frame_rate)
    df = t.df.reindex(posts, method='bfill', limit=1)
    for c in df.columns:
        if not c.startswith('marker'):
            df[c] = df[c].bfill().ffill()
        elif c[-1] in 'xyz':
            df[c] = df[c].interpolate(limit=max_interpolate)
        elif c[-1] in 'c':
            df[c] = df[c].fillna(value=1.2345, limit=max_interpolate)
    t.df = df

    t.save(t.root.replace(root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    frame_rate=('reindex frames to this rate', 'option', None, float),
    interpolate=('interpolate gaps of this many frames', 'option', None, int),
    max_speed=('treat frames with > speed as dropouts', 'option', None, float),
    max_acc=('treat frames with > acc as dropouts', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, interpolate=3, max_speed=5, max_acc=100):
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    work = joblib.delayed(reindex)
    args = root, output, frame_rate, interpolate, max_speed, max_acc
    joblib.Parallel(-1)(work(t, *args) for t in trials)


if __name__ == '__main__':
    climate.call(main)
