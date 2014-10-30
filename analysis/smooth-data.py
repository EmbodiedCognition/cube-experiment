import climate

import database


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    frame_rate=('reindex frames to this rate', 'option', None, float),
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    frames=('number of frames for SVT', 'option', None, int),
)
def main(root, output, frame_rate=100., accuracy=0.002, threshold=100, frames=5):
    for t in database.Experiment(root).trials_matching('*'):
        t.reindex(frame_rate=frame_rate)
        t.svt(threshold, accuracy, frames)
        t.save(t.root.replace(root, output))


if __name__ == '__main__':
    climate.call(main)
