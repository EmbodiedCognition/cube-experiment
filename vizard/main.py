from __future__ import print_function

# python imports
import datetime
import gzip
import logging
import os
import random
import shutil

# vizard imports
import viz
import vizproximity
import vizshape
import viztask

# local imports
import vrlab
import suit
import targets

# module constants
TIMESTAMP_FORMAT = '%Y%m%d%H%M%S'
BASE_PATH = 'C:\\Documents and Settings\\vrlab\\Desktop\\target-data'


class Trial(vrlab.Trial):
    '''Manage a single trial of the target-reaching experiment.'''

    def __init__(self, block, targets, **kwargs):
        super(Trial, self).__init__()

        self.block = block
        self.home = targets[0]
        self.targets = targets[1:]

        self.current_target = self.previous_target = self.home

        self.suit = block.experiment.suit
        self.records = []
        self._timer = self.add_periodic(1. / 100, self.record_data)

    @property
    def index(self):
        if os.path.isdir(self.block.output):
            return len(os.listdir(self.block.output))
        return 0

    def record_data(self):
        self.records.append((
            viz.tick() - self.start_tick,
            self.previous_target,
            self.current_target,
            self.suit.get_markers(),
        ))

    def wait_for_touch(self, target):
        target.activate(self.block.experiment.prox)
        #yield viztask.waitKeyDown(' ')
        yield target.signal.wait()

    def target_sequence(self):
        raise NotImplementedError

    def setup(self):
        yield self.wait_for_touch(self.home)
        self.start_tick = viz.tick()

    def run(self):
        for target in self.target_sequence():
            self.previous_target = self.current_target
            self.current_target = target
            yield self.wait_for_touch(target)
            target.sphere.color(viz.WHITE)

    def teardown(self):
        vrlab.sounds.cowbell.play()
        self.write_records()

    def trial_description(self):
        return '{}{:02d}'.format(self.TRIAL_TYPE, self.index)

    def write_records(self):
        stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        output = os.path.join(
            self.block.output,
            '{}-{}.csv.gz'.format(stamp, self.trial_description()))

        # open file and define helper to write data
        handle = gzip.open(output, 'w')
        def w(msg, *args, **kwargs):
            handle.write(msg.format(*args, **kwargs))

        # write csv file header
        w('time,')
        w('source,source-x,source-y,source-z,')
        w('target,target-x,target-y,target-z,')
        w('effector,effector-x,effector-y,effector-z,effector-c')
        for label in suit.MARKER_LABELS:
            w(',{0}-x,{0}-y,{0}-z,{0}-c', label)
        w('\n')

        # write data frames
        eff = self.block.effector
        for elapsed, prev, curr, frame in self.records:
            w('{}', elapsed)
            w(',{t.index},{t.center[0]},{t.center[1]},{t.center[2]}', t=prev)
            w(',{t.index},{t.center[0]},{t.center[1]},{t.center[2]}', t=curr)
            w(',{i},{m.pos[0]},{m.pos[1]},{m.pos[2]},{m.cond}', i=eff, m=frame[eff])
            for i in range(len(frame)):
                w(',{m.pos[0]},{m.pos[1]},{m.pos[2]},{m.cond}', m=frame[i])
            w('\n')

        # finish up
        handle.close()
        logging.info('wrote %d records to trial output %s',
                     len(self.records), os.path.basename(output))


class HubAndSpokeTrial(Trial):
    TRIAL_TYPE = 'hubspoke'

    def target_sequence(self):
        for target in self.targets:
            yield target
            yield self.home


class CircuitTrial(Trial):
    TRIAL_TYPE = 'circuit'

    def __init__(self, block, targets, circuit=0):
        super(CircuitTrial, self).__init__(block, targets)

        self.circuit_num = circuit

    def trial_description(self):
        return '{}{:02d}'.format(self.TRIAL_TYPE, self.circuit_num)

    def target_sequence(self):
        for target in self.targets:
            yield target


class ControlTrial(HubAndSpokeTrial):
    TRIAL_TYPE = 'control'
    TARGETS = [targets.NUMBERED[i] for i in (7, 8, 3)]

    def __init__(self, block, targets, home=targets.NUMBERED[6]):
        random.shuffle(self.TARGETS)
        assert home.index in (6, 10, 9)
        super(ControlTrial, self).__init__(block, [home] + self.TARGETS)


class Block(vrlab.Block):
    '''Manage a block of trials in the tracing experiment.

    This class handles block setup (playing a sound, making a directory for
    recording data) and generates trials in the block by choosing sequences of
    targets.
    '''

    def __init__(self,
                 experiment,
                 effector,
                 trial_factory,
                 num_trials,
                 ):
        super(Block, self).__init__()

        self.experiment = experiment
        self.effector = effector
        self.trial_factory = trial_factory
        self.num_trials = num_trials

        stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.output = os.path.join(
            experiment.output, '{}-block{:02d}'.format(stamp, self.index))

        logging.info('NEW BLOCK -- effector %s, trials %s',
                     self.effector, self.trial_factory.__name__)

    @property
    def index(self):
        if os.path.isdir(self.experiment.output):
            return len(os.listdir(self.experiment.output))
        return 0

    def setup(self):
        self.experiment.prox.addTarget(self.experiment.leds[self.effector])

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        yield viztask.waitKeyDown(' ')

    def teardown(self):
        self.experiment.prox.clearTargets()

    def generate_trials(self):
        idx = list(range(len(targets.CIRCUITS)))
        #random.shuffle(idx)
        for i in idx[:self.num_trials]:
            targets = [targets.NUMBERED[t] for t in targets.CIRCUITS[i]]
            yield self.trial_factory(self, targets, circuit=i)


class Experiment(vrlab.Experiment):
    '''Manage a series of blocks in the reaching experiment.

    This class handles global experiment setup for a single subject. To set up,
    we turn on the motion-capture thread, create some experiment-relevant Vizard
    objects for representing the mocap leds and targets, and create a virtual
    environment for visualization.
    '''

    def find_output(self, threshold_min):
        '''Locate an output directory for a subject.

        This method looks at existing output directories and reuses an existing
        directory if one was modified in the past "threshold" minutes. If no
        such directory exists, it creates a new one.
        '''
        now = datetime.datetime.now()
        for bn in os.listdir(BASE_PATH):
            stamp, _ = bn.split('-')
            then = datetime.datetime.strptime(stamp, TIMESTAMP_FORMAT)
            if now - then < datetime.timedelta(seconds=60 * threshold_min):
                return bn
        return '{}-{:08x}'.format(now.strftime(TIMESTAMP_FORMAT),
                                  random.randint(0, 0xffffffff))

    def setup(self):
        # set up a folder to store data for a subject.
        dirname = self.find_output(threshold_min=7)
        self.output = os.path.join(BASE_PATH, dirname)
        logging.info('storing output in %s', self.output)

        # configure the phasespace.
        mocap = vrlab.Phasespace('192.168.1.230', freq=120)
        self.suit = mocap.track_points(range(len(suit.MARKER_LABELS)))
        self.leds = []
        for i in range(50):
            sphere = vizshape.addSphere(0.02, color=viz.RED)
            self.suit.link_marker(i, sphere)
            self.leds.append(sphere)
        mocap.start_thread()

        # set up a proximity manager to detect touch events.
        self.prox = vizproximity.Manager()
        #self.prox.setDebug(viz.ON)

        # add an environment and navigation to help with visualization.
        self.environment = viz.addChild('dojo.osgb')
        viz.cam.setHandler(None)

    def generate_blocks(self):
        #yield Block(self, effector=suit.MARKERS.R_FING_INDEX, trial_factory=ControlTrial, num_trials=10)
        #yield Block(self, effector=suit.MARKERS.R_FING_INDEX, trial_factory=HubAndSpokeTrial, num_trials=5)
        yield Block(self, effector=suit.MARKERS.R_FING_INDEX, trial_factory=CircuitTrial, num_trials=6)


if __name__ == '__main__':
    logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG,
            format='%(levelname).1s %(asctime)s %(name)s:%(lineno)d %(message)s',
        )
    Experiment().main(fullscreen=False)
