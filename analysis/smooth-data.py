#!/usr/bin/env python

import climate
import collections
import joblib

import database


def smooth(args):
    t = args.trial
    t.load()
    t.reindex(args.frame_rate)
    t.mask_dropouts()
    t.svt(threshold=args.threshold,
          max_rmse=args.accuracy,
          consec_frames=args.frames,
          log_every=0)
    t.lowpass(args.lowpass, only_dropouts=False)
    t.save(t.root.replace(args.root, args.output))


Args = collections.namedtuple('Args', 'trial root output frame_rate accuracy threshold frames lowpass')

@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    frame_rate=('reindex frames to this rate', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    frames=('number of frames for SVT', 'option', None, int),
    lowpass=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, frame_rate=100., accuracy=0.002, threshold=500, frames=3, lowpass=10.):
    args = root, output, frame_rate, accuracy, threshold, frames, lowpass
    trials = database.Experiment(root).trials_matching('*', load=False)
    proc = joblib.delayed(smooth)
    joblib.Parallel(-1)(proc(Args(t, *args) for t in trials))


if __name__ == '__main__':
    climate.call(main)
