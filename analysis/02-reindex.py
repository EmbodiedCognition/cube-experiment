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
    #'marker04-r-head-mid',
    #'marker05-l-head-mid',
    'marker06-r-collar',
    'marker07-r-shoulder',
    'marker08-r-elbow',
    'marker09-r-wrist',
    #'marker10-r-fing-pinky',
    #'marker11-r-fing-ring',
    #'marker12-r-fing-middle',
    'marker13-r-fing-index',
    'marker14-r-mc-outer',
    #'marker15-r-mc-inner',
    'marker16-r-thumb-base',
    #'marker17-r-thumb-tip',
    'marker18-l-collar',
    'marker19-l-shoulder',
    'marker20-l-elbow',
    'marker21-l-wrist',
    #'marker22-l-fing-pinky',
    #'marker23-l-fing-ring',
    #'marker24-l-fing-middle',
    'marker25-l-fing-index',
    'marker26-l-mc-outer',
    #'marker27-l-mc-inner',
    'marker28-l-thumb-base',
    #'marker29-l-thumb-tip',
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

SPEEDS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
ACCELS = (0.5, 1, 2, 5, 10, 20, 50, 100, 200)

def reindex(t, root, output, frame_rate, interpolate, max_speed, max_acc):
    t.load()

    def below(s, r):
        return '{}<{}'.format((s < r).sum(), r)

    # mask marker frames that are dropouts or are moving too fast.
    ds = pd.Series(t.df.index, index=t.df.index)
    dt = ds.shift(-1) - ds.shift(1)
    for marker in t.marker_columns:
        c = t.df[marker + '-c']
        x = t.trajectory(marker)
        v = x.diff(2).shift(-1).div(dt, axis='index')
        a = v.diff(2).shift(-1).div(dt, axis='index')
        spd = np.sqrt((v * v).sum(axis=1))
        acc = np.sqrt((a * a).sum(axis=1))
        mask = (c < 0) | (spd > max_speed) | (acc > max_acc)
        logging.info(
            '%s %s %s %s: %d frames, masking %d = %d dropouts + %d spd (%s) + %d acc (%s)',
            t.subject.key, t.block.key, t.key, marker, len(t.df), mask.sum(), (c < 0).sum(),
            (spd > max_speed).sum(), ' '.join(below(spd, r) for r in SPEEDS),
            (acc > max_acc).sum(), ' '.join(below(acc, r) for r in ACCELS))
        t.df.ix[mask, marker + '-x':marker + '-c'] = float('nan')

    # drop unusable marker data.
    for c in t.columns:
        if c.startswith('marker') and c[:-2] not in MARKERS:
            del t.df[c]

    # reindex to regularly-spaced temporal index values.
    posts = np.arange(0, t.df.index[-1], 1. / frame_rate)
    df = t.df.reindex(posts, method='bfill', limit=1)
    for c in df.columns:
        if not c.startswith('marker'):
            df[c] = df[c].bfill().ffill()
        elif c[-1] in 'xyz':
            df[c] = df[c].interpolate(limit=interpolate)
        elif c[-1] in 'c':
            df[c] = df[c].fillna(value=1.2345, limit=interpolate)
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
