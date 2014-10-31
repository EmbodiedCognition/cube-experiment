import climate

import database


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
    for t in database.Experiment(root).trials_matching('*'):
        t.replace_dropouts(visibility)
        t.reindex(frame_rate)
        t.svt(threshold, accuracy, frames)
        t.lowpass(lowpass, only_dropouts=True)
        t.svt(threshold, accuracy, frames)
        t.lowpass(lowpass, only_dropouts=False)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
