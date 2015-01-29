import climate

import source


def main():
    exp = source.Experiment('/home/leif/src/cube-experiment/data')
    t = exp.subjects[0].blocks[0].trials[0]
    t.load()
    print(
        t.target_contact_frames,
        t.df.source.iloc[t.target_contact_frames],
        t.df.target.iloc[t.target_contact_frames],
        t.df.target.iloc[(t.target_contact_frames + 1)[:-1]],
    )
    t.realign()
    print(
        t.target_contact_frames,
        t.df.source.iloc[t.target_contact_frames],
        t.df.target.iloc[t.target_contact_frames],
        t.df.target.iloc[(t.target_contact_frames + 1)[:-1]],
    )


if __name__ == '__main__':
    climate.call(main)
