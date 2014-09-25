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
        self.center = x, y, z
        self.sphere = vizshape.addSphere(0.2)
        self.sphere.setPosition(self.center)
        self.sphere.color(viz.WHITE)
        self.sensor = vizproximity.addBoundingSphereSensor(self.sphere, scale=1)
        self.signal = viztask.Signal()
        self.sound = viz.addAudio('{:02d}.wav'.format(index))

    def activate(self, prox):
        prox.clearSensors()

        prox.addSensor(self.sensor)

        prox.onEnter(self.sensor, lambda e: vrlab.sounds.drip.play())
        prox.onEnter(self.sensor, lambda e: self.sphere.color(viz.BLUE))
        prox.onEnter(self.sensor, self.signal.send)

        prox.onExit(self.sensor, lambda e: self.sphere.color(viz.WHITE))


NUMBERED = (
    Target( 0, -1.98, 0.05, -1.86),
    Target( 1, -1.72, 1.83,  2.26),
    Target( 2,  0.00, 0.05,  1.86),
    Target( 3,  1.73, 0.05, -1.79),
    Target( 4,  1.89, 0.99,  2.26),
    Target( 5, -2.14, 0.93,  0.10),
    Target( 6, -0.24, 0.90, -1.76),
    Target( 7,  1.51, 1.81, -1.76),
    Target( 9,  1.79, 0.05,  0.00),
    Target(10,  0.10, 1.89,  0.10),
    Target(11, -0.24, 1.86,  2.26),
)


CIRCUITS = (
    (10, 0, 1, 3, 8, 4, 11, 7, 9, 6, 5, 2),
    (7, 1, 0, 11, 9, 2, 8, 3, 6, 4, 10, 5),
    (3, 0, 8, 11, 5, 10, 6, 1, 4, 2, 9, 7),
    (11, 8, 7, 3, 4, 6, 9, 5, 0, 2, 1, 10),
    (4, 7, 8, 5, 6, 0, 3, 1, 9, 10, 2, 11),
    (10, 3, 9, 1, 2, 4, 5, 7, 11, 0, 6, 8),
)
