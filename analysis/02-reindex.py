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
    #'marker16-r-thumb-base',
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
    #'marker28-l-thumb-base',
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


def reindex(t, root, output, frame_rate, max_speed, max_acc):
    t.load()

    # drop marker data not in MARKERS.
    for c in set(t.marker_columns) - set(MARKERS):
        del t.df[c + '-x']
        del t.df[c + '-y']
        del t.df[c + '-z']
        del t.df[c + '-c']

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
        t.log('%s %d frames, masking %d = %d dropouts + %d spd (%s) + %d acc (%s)',
              marker, len(t.df), mask.sum(), (c < 0).sum(),
              (spd > max_speed).sum(), ' '.join(below(spd, r) for r in SPEEDS),
              (acc > max_acc).sum(), ' '.join(below(acc, r) for r in ACCELS))
        t.df.ix[mask, marker + '-x':marker + '-c'] = float('nan')

    # fill nans locally before reindexing.
    limit = int(np.ceil(100 / frame_rate))
    for c in t.df.columns:
        s, n = t.df[c], None
        if not c.startswith('marker'):
            # fill all nans in non-marker columns (e.g., source, target).
            n = s.bfill().ffill()
        elif c[-1] in 'xyz':
            # interpolate nearby position data.
            fw = s.interpolate(method='values', limit=limit)
            bw = s[::-1].interpolate(method='values', limit=limit)
            n = fw.fillna(bw[::-1])
        elif c[-1] in 'c':
            # fill nearby condition values.
            fw = s.fillna(value=1.2345, limit=limit)
            bw = s[::-1].fillna(value=1.2345, limit=limit)
            n = fw.fillna(bw[::-1])
        t.df[c] = n

    # reindex to regularly-spaced temporal index values.
    t.df = t.df.reindex(np.arange(0, t.df.index[-1], 1 / frame_rate),
                        method='nearest', limit=1)

    t.save(t.root.replace(root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    frame_rate=('reindex frames to this rate', 'option', None, float),
    max_speed=('treat frames with > speed as dropouts', 'option', None, float),
    max_acc=('treat frames with > acc as dropouts', 'option', None, float),
)
def main(root, output, pattern='*', frame_rate=100, max_speed=5, max_acc=100):
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    args = root, output, frame_rate, max_speed, max_acc
    work = joblib.delayed(reindex)
    joblib.Parallel(-1)(work(t, *args) for t in trials)


if __name__ == '__main__':
    climate.call(main)
