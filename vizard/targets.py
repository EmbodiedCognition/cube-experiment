import viz
import vizproximity
import vizshape
import viztask

import vrlab


class Target:
    '''A target is a single cube in the motion-capture space.

    Subjects are tasked with touching the cubes during the experiment.
    '''

    def __init__(self, index, x, y, z):
        self.index = index
        self.center = x, y, z
        self.sphere = vizshape.addSphere(0.1)
        self.sphere.setPosition(self.center)
        self.sphere.color(viz.WHITE)
        self.sensor = vizproximity.addBoundingSphereSensor(self.sphere, scale=1)
        self.signal = viztask.Signal()
        self.sound = viz.addAudio('{:02d}.wav'.format(index))

    def __str__(self):
        return 'Target({:2d}, {:5.2f}, {:5.2f}, {:5.2f})'.format(
            self.index, *self.center)

    def activate(self, prox):
        prox.clearSensors()

        prox.addSensor(self.sensor)

        prox.onEnter(self.sensor, lambda e: vrlab.sounds.drip.play())
        prox.onEnter(self.sensor, lambda e: self.sphere.color(viz.BLUE))
        prox.onEnter(self.sensor, self.signal.send)

        prox.onExit(self.sensor, lambda e: self.sphere.color(viz.WHITE))

        self.sphere.color(viz.GREEN)
        self.sound.play()


NUMBERED = (
    # dirs:   "right" "up" "forward"
    Target( 0, -1.82, 0.05, -1.96),
    Target( 1, -1.70, 1.82,  2.18),
    Target( 2, -0.02, 0.05,  1.78),
    Target( 3,  1.88, 0.05, -1.82),
    Target( 4,  1.94, 0.93,  2.26),
    Target( 5, -1.98, 0.97,  0.08),
    Target( 6, -0.10, 1.05, -1.80),
    Target( 7,  1.71, 1.88, -1.76),
    Target( 8,  2.00, 1.04, -1.73),
    Target( 9,  1.83, 0.05,  0.02),
    Target(10,  0.14, 1.90,  0.10),
    Target(11, -0.18, 1.86,  2.21),
)


CIRCUITS = (
    (10,  0,  1,  3,  8,  4, 11,  7,  9,  6,  5,  2),
    ( 7,  1,  0, 11,  9,  2,  8,  3,  6,  4, 10,  5),
    ( 3,  0,  8, 11,  5, 10,  6,  1,  4,  2,  9,  7),
    (11,  8,  7,  3,  4,  6,  9,  5,  0,  2,  1, 10),
    ( 4,  7,  8,  5,  6,  0,  3,  1,  9, 10,  2, 11),
    (10,  3,  9,  1,  2,  4,  5,  7, 11,  0,  6,  8),

    (10,  0,  6,  3, 11,  5,  8,  1,  4,  9,  7,  2),
    ( 6, 10,  4,  2,  9,  0,  8,  7,  5,  1, 11,  3),
    ( 8,  0, 10, 11,  5,  4,  7,  3,  2,  6,  1,  9),
    ( 3,  0,  6,  8,  1,  5,  2,  9, 11,  7,  4, 10),
    ( 5,  7,  9,  3, 10, 11,  2,  1,  4,  8,  6,  0),
    ( 0,  3, 10,  6,  4,  5,  7,  9,  1,  2, 11,  8),
    ( 5,  1, 11,  3,  0,  2,  7, 10,  6,  8,  9,  4),
    (11, 10,  4,  1,  3,  9,  5,  6,  2,  8,  0,  7),
    ( 6,  1,  2,  3,  0,  9,  5, 10,  8,  7, 11,  4),
    ( 9,  4,  8,  7,  2,  0,  1, 10,  5,  3, 11,  6),
    ( 0,  3,  8,  4,  9,  6, 10,  7,  1,  2,  5, 11),
    ( 1, 10,  3,  4,  0,  2,  5,  9,  8, 11,  6,  7),
    ( 7, 11,  8,  3,  6,  5,  2,  1,  0,  4, 10,  9),
    ( 6,  1,  5,  0, 11,  2,  8, 10,  7,  9,  3,  4),
    ( 0, 10,  1,  7,  5,  8, 11,  2,  4,  6,  9,  3),
    ( 9,  1, 10,  3,  7, 11,  8,  0,  5,  4,  6,  2),
    ( 1,  6,  7,  2,  0,  5, 10, 11,  3,  9,  8,  4),
    (11,  9,  1,  7,  0,  4,  3,  2, 10,  5,  8,  6),
    ( 1,  7,  3, 10,  2,  8,  4, 11,  9,  0,  6,  5),
    ( 4, 10,  1,  8,  3,  5,  9,  0, 11,  7,  6,  2),
    ( 7,  8,  2,  0,  9,  4,  1,  3,  5,  6, 11, 10),
    ( 6, 11,  7,  8,  9, 10,  0,  2,  4,  5,  3,  1),
    ( 7,  5,  0, 10,  8,  2, 11,  1,  9,  6,  3,  4),
    (11,  2,  3,  8,  1,  5,  7,  6,  9, 10,  4,  0),
    ( 3,  8,  5, 11, 10,  6,  4,  2,  7,  0,  1,  9),
    ( 5, 11,  6,  8, 10,  0,  1,  3,  7,  4,  9,  2),
    ( 9,  8,  5,  6,  1,  0,  3,  7, 10,  2,  4, 11),
    ( 5,  0,  7,  3,  2, 10,  1,  6,  9, 11,  4,  8),
    ( 3,  9,  7, 10,  8,  6, 11,  0,  5,  4,  2,  1),
    (10,  3,  5,  9,  6,  4,  7,  0,  8,  2, 11,  1),
    ( 9, 11,  0, 10,  4,  8,  5,  2,  7,  6,  3,  1),
    ( 1,  4,  7,  8,  2, 10,  9,  5,  3,  6,  0, 11),
    ( 0,  7,  1,  8, 10,  9,  2,  6,  5,  4,  3, 11),
    ( 0,  4,  6, 10,  2,  9,  7,  8,  3,  1, 11,  5),
    ( 9,  2,  6,  0,  3, 11,  4,  5, 10,  7,  1,  8),
    ( 7,  4, 11,  1,  0,  8,  9, 10,  5,  2,  3,  6),
)
