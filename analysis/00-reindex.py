#!/usr/bin/env python

import climate
import joblib
import lmj.cubes

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


def reindex(t, root, output, frame_rate, interpolate):
    t.load()
    t.mask_dropouts()
    t.reindex(frame_rate, interpolate)
    for c in t.columns:
        if c.startswith('marker') and c[:-2] not in MARKERS:
            del t.df[c]
    t.save(t.root.replace(root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    pattern=('process only trials matching this pattern', 'option'),
    frame_rate=('reindex frames to this rate', 'option', None, float),
    interpolate=('interpolate gaps of this many frames', 'option', None, int),
)
def main(root, output, pattern='*', frame_rate=100, interpolate=33):
    trials = lmj.cubes.Experiment(root).trials_matching(pattern)
    work = joblib.delayed(reindex)
    args = root, output, frame_rate, interpolate
    joblib.Parallel(-1)(work(t, *args) for t in trials)


if __name__ == '__main__':
    climate.call(main)
