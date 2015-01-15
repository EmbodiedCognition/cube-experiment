#!/usr/bin/env python

import climate
import collections
import multiprocessing as mp

import database


def smooth(args):
    args.trial.load()
    args.trial.replace_dropouts(args.visibility)
    args.trial.reindex(args.frame_rate)
    args.trial.svt(args.threshold, 4 * args.accuracy, args.frames)
    args.trial.lowpass(1 * args.lowpass, only_dropouts=True)
    args.trial.svt(args.threshold, 2 * args.accuracy, args.frames)
    args.trial.lowpass(2 * args.lowpass, only_dropouts=True)
    args.trial.svt(args.threshold, 1 * args.accuracy, args.frames)
    args.trial.lowpass(4 * args.lowpass, only_dropouts=False)
    args.trial.save(args.trial.root.replace(args.root, args.output))


Args = collections.namedtuple('Args', 'trial root output frame_rate visibility accuracy threshold frames lowpass')

@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    frame_rate=('reindex frames to this rate', 'option', None, float),
    visibility=('drop markers with < this ratio of visible frames', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    frames=('number of frames for SVT', 'option', None, int),
    lowpass=('lowpass filter at N Hz', 'option', None, float),
)
def main(root, output, frame_rate=100., visibility=0.1, accuracy=0.001, threshold=100, frames=3, lowpass=5.):
    args = root, output, frame_rate, visibility, accuracy, threshold, frames, lowpass
    trials = database.Experiment(root).trials_matching('*', load=False)
    mp.Pool().map(smooth, (Args(t, *args) for t in trials))


if __name__ == '__main__':
    climate.call(main)
