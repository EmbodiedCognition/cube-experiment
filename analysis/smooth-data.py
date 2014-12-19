import climate
import multiprocessing as mp

import database


def smooth(args):
    trial, root, output, frame_rate, visibility, accuracy, threshold, frames, lowpass = args
    trial.replace_dropouts(visibility)
    trial.reindex(frame_rate)
    trial.svt(threshold, 4 * accuracy, frames)
    trial.lowpass(lowpass, only_dropouts=True)
    trial.svt(threshold, 2 * accuracy, frames)
    trial.lowpass(lowpass, only_dropouts=True)
    trial.svt(threshold, accuracy, frames)
    trial.lowpass(lowpass, only_dropouts=False)
    trial.save(trial.root.replace(root, output))


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    frame_rate=('reindex frames to this rate', 'option', None, float),
    visibility=('drop markers with < this ratio of visible frames', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    frames=('number of frames for SVT', 'option', None, int),
    lowpass=('lowpass filter at N Hz after SVT', 'option', None, float),
)
def main(root, output, frame_rate=100., visibility=0.1, accuracy=0.001, threshold=100, frames=5, lowpass=10.):
    args = root, output, frame_rate, visibility, accuracy, threshold, frames, lowpass
    trials = database.Experiment(root).trials_matching('*')
    mp.Pool().map(smooth, ((t, ) + args for t in trials))


if __name__ == '__main__':
    climate.call(main)
