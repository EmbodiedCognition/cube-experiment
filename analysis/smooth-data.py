import climate

import database


@climate.annotate(
    root='load data files from this directory tree',
    output='save smoothed data files to this directory tree',
    accuracy=('fit SVT with this accuracy', 'option', None, float),
    threshold=('SVT threshold', 'option', None, float),
    frames=('number of frames for SVT', 'option', None, int),
)
def main(root, output, accuracy=0.005, threshold=1000, frames=5):
    for trial in database.Experiment(root).trials_matching('*'):
        t.reindex()
        t.svt(threshold, accuracy, frames)
        t.save(t.root.replace(root, output))



if __name__ == '__main__':
    climate.call(main)
