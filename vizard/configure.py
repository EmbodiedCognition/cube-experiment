#!/usr/bin/env python

import sys
import time

import viz
import vizshape
import viztask

import suit
import targets
import vrlab

M = [13, 12, 11, 10]
TRACKER = None

def workflow():
    vrlab.sounds.cowbell.play()
    yield viztask.waitTime(5)
    for target in targets.NUMBERED:
        if True:# target.center == (0, 0, 0):
            target.sound.play()
            yield viztask.waitTime(10)
            vrlab.sounds.drip.play()
            x = y = z = n = 0
            for _ in range(10):
                for m in M:
                    t = TRACKER.get_marker(m)
                    if t.cond != -1:
                        xi, yi, zi = t.pos
                        x += xi
                        y += yi
                        z += zi
                        n += 1
                time.sleep(0.01)
            if n == 0:
                n = 1
                vrlab.sounds.cowbell.play()
            target.center = x / n, y / n, z / n
            print('    {},'.format(target))


def main():
    global TRACKER

    viz.go(0)

    # configure the phasespace.
    mocap = vrlab.Phasespace('192.168.1.230', freq=100., postprocess=True)
    mocap.start_thread()
    TRACKER = mocap.track_points(M)

    viztask.schedule(workflow)


if __name__ == '__main__':
    main()
