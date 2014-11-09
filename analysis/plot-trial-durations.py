import climate
import collections
import lmj.plot
import numpy as np

import database
import plots


@climate.annotate(
    root='load experiment data from this directory',
    pattern=('plot data from files matching this pattern', 'option'),
)
def main(root, pattern='*.csv.gz'):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in database.Experiment(root).subjects:
        for i, b in enumerate(s.blocks):
            for j, t in enumerate(b.trials):
                if t.matches(pattern):
                    t.load()
                    data[i][j].append(t.df.index[-1] / t.total_distance)
    counts = []
    means = []
    stds = []
    labels = []
    for i in sorted(data):
        for j in sorted(data[i]):
            counts.append(len(data[i][j]))
            means.append(np.mean(data[i][j]))
            stds.append(np.std(data[i][j]))
            labels.append('{}/{}'.format(i + 1, j + 1))
    xs = np.arange(len(means))
    means = np.array(means)
    stds = np.array(stds)
    with plots.plot() as ax:
        ax.plot(xs, means, color='#111111')
        ax.fill_between(xs, means - stds, means + stds,
                        color='#111111', alpha=0.7, lw=0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)


if __name__ == '__main__':
    climate.call(main)
