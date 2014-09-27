import climate
import lmj.plot
import source


def main(subject):
    subj = source.Subject(subject)
    ax = lmj.plot.axes(111, projection='3d', aspect='equal')
    for i, block in enumerate(subj.blocks):
        trial = block.trials[0]
        trial.load()
        df = trial.marker_trajectory('r-fing-index')
        ax.plot(df.x, df.z, zs=df.y, color=(i / len(subj.blocks), 0, 0), alpha=0.9)
    lmj.plot.show()



if __name__ == '__main__':
    climate.call(main)

