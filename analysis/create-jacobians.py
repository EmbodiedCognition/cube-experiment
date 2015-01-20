import climate

import database


def main(root, pattern='*'):
    for trial in database.Experiment(root).trials_matching(pattern, load=False):
        trial.load()

        body = database.Movement(trial.df.copy())
        body.make_body_relative()
        body.add_velocities()

        goal = database.Movement(trial.df.copy())
        tgt = goal.target_trajectory
        goal.recenter(tgt.x, tgt.y, tgt.z)
        goal.add_velocities()

        jac = pd.DataFrame([], index=trial.df.index)
        for cb in body.marker_channel_columns:
            for cg in body.marker_channel_columns:
                jac['body-{}/goal-{}'.format(cb, cg)] = body[cb] / goal[cg]

        break


if __name__ == '__main__':
    climate.call(main)
