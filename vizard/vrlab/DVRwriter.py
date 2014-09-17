import viz
import dvr_arr as dvr


class DVRwriter(viz.EventClass):
    def __init__(self, filename, metadata, viewport, noeye=False):
        viz.EventClass.__init__(self)

        self.callback(viz.EXIT_EVENT, self.cleanup)
        self.metadata = metadata
        self.dvr = dvr.create(1)
        dvr.open(self.dvr, filename, viewport, noeye)

    def cleanup(self):
        dvr.close(self.dvr, self.metadata)

    def turnOff(self):
        dvr.stop(self.dvr)

    def turnOn(self):
        dvr.start(self.dvr)

    def write(self, writables):
        lst = []
        for w in writables:
            if w is not None:
                output = w.getOutput()
                if output:
                    lst.append(output)
        dvr.post(self.dvr, '\t'.join(lst) + '\n')
