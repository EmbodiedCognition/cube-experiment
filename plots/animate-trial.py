#!/usr/bin/env python

import climate
import collections
import lmj.cubes
import pagoda.viewer

BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (0.9, 0.2, 0.2)
YELLOW = (0.9, 0.9, 0.2)
ORANGE = (0.9, 0.6, 0.3)
GREEN = (0.2, 0.9, 0.2)
BLUE = (0.2, 0.3, 0.9)
COLORS = (WHITE, RED, YELLOW, GREEN, BLUE, ORANGE)


class Window(pagoda.viewer.Window):
    def __init__(self, trial, paused=False):
        super().__init__(paused=paused)

        values = trial.df[[c for c in trial.columns if c.startswith('marker')]].values
        values = values.reshape((len(trial.df), -1, 4))
        self._frames = iter(values)
        self._frame_rate = 1 / trial.approx_delta_t

        self.targets = trial.target_trajectory.drop_duplicates().values

        self._maxlen = 4
        self._trails = [[] for _ in range(len(trial.marker_columns))]
        self._reset_trails()

    def grab_key_press(self, key, modifiers, keymap):
        if key == keymap.PLUS or key == keymap.EQUAL:
            self._maxlen *= 2
            self._reset_trails()
            return True
        if key == keymap.UNDERSCORE or key == keymap.MINUS:
            self._maxlen = max(1, self._maxlen // 2)
            self._reset_trails()
            return True
        if key == keymap.RIGHT:
            skip = int(self._frame_rate)
            if modifiers:
                skip *= 10
            [self._next_frame() for _ in range(skip)]
            return True

    def render(self, dt):
        for x, y, z in self.targets:
            self.draw_sphere(color=BLACK + (0.7, ),
                             translate=(x, z, y),
                             scale=(0.1, 0.1, 0.1))

        for t, trail in enumerate(self._trails):
            if not len(trail):
                continue
            self.set_color(*(COLORS[t % len(COLORS)] + (0.7, )))
            self.draw_lines(t for t in trail if t is not None)
            if trail[-1] is not None:
                self.draw_sphere(translate=trail[-1], scale=(0.02, 0.02, 0.02))

    def _reset_trails(self):
        self._trails = [collections.deque(t, self._maxlen) for t in self._trails]

    def _next_frame(self):
        try:
            return next(self._frames)
        except StopIteration:
            pyglet.app.exit()

    def step(self, dt):
        for trail, (x, y, z, c) in zip(self._trails, self._next_frame()):
            trail.append((x, z, y) if 10 > c > 0 else None)


def main(root, pattern):
    for t in lmj.cubes.Experiment(root).trials_matching(pattern):
        t.load()
        #t.add_velocities(smooth=0)
        Window(t).run()


if __name__ == '__main__':
    climate.call(main)
