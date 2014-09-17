import viz
import vizmat
import vizact


class Camera(viz.EventClass):
    def __init__(self,
                 offset,
                 particle=None,
                 sensorNum=0,
                 attachTo=viz.MainView,
                 preTrans=[0, 0, 0.1778]):
        viz.EventClass.__init__(self)

        self.offset = offset
        self.particle = particle
        self.sensorNum = sensorNum
        self.attachTo = attachTo
        self.preTran = preTrans

        self.off = True

        self.vrpn = viz.add('vrpn6.dle')
        self.tracker = self.vrpn.addTracker('Tracker0@192.168.1.6', sensorNum)
        self.tracker.swapPos([1, 3, 2])
        self.tracker.swapQuat([-1, -3, -2, 4])

        self.preMat = vizmat.Transform()
        self.preMat.preEuler([-90, 0, 0])
        self.preMat.preTrans(self.preTran)

        self.postMat = vizmat.Transform()
        self.postMat.postTrans(offset)

        vizact.ontimer(0, self.updateView)
        self.pos = [0, 0, 0]
        self.rot = [0, 0, 0, 0]
        self.turnOn()

    def __del__(self):
        self.quit()

    def turnOff(self):
        self.setEnabled(False)
        self.off = True

    def turnOn(self):
        self.off = False
        self.setEnabled(True)

    def isOn(self):
        return not self.off

    def quit(self):
        print "Quitting"

    def getOutput(self):
        return ' HiBall{}: {} {}'.format(self.sensorNum, self.pos, self.rot)

    def updateView(self):
        data = self.tracker.getData()
        self.pos = data[:3]
        self.rot = data[3:7]
        self.m = viz.Transform()
        self.m.setQuat(self.rot)
        self.m.postTrans(self.pos)
        self.m.postAxisAngle(0, 1, 0, 90)
        self.m.preMult(self.preMat)
        self.m.postMult(self.postMat)

        if self.attachTo != None and not self.off:
            self.attachTo.setMatrix(self.m)

        if self.particle != None:
            pos = self.m.getTrans()
            self.particle.moveTo([pos[0], self.offset[1], pos[2]])
