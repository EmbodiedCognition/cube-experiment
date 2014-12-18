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
GAP_MINUTES = 50
TIMESTAMP_FORMAT = '%Y%m%d%H%M%S'
BASE_PATH = 'C:\\Documents and Settings\\vrlab\\Desktop\\target-data'


class Trial(vrlab.Trial):
    '''Manage a single trial of the target-reaching experiment.'''

    def __init__(self, block, targets):
        super(Trial, self).__init__()

        self.block = block
        self.home = targets[0]
        self.targets = targets[1:]

        self.trial_label = ''.join('{:x}'.format(t.index) for t in targets)

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
        for target in self.targets:
            yield target

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

    def write_records(self):
        effector = suit.MARKER_LABELS[self.block.effector]

        stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        output = os.path.join(
            self.block.output,
            '{}-trial{:02d}-{}.csv.gz'.format(stamp, self.index, self.trial_label))

        # open file and define helper to write data
        handle = gzip.open(output, 'w')
        def w(msg, *args, **kwargs):
            handle.write(msg.format(*args, **kwargs))

        # write csv file header
        w('time,effector')
        w(',source,source-x,source-y,source-z')
        w(',target,target-x,target-y,target-z')
        headers = ''.join(',marker{i:02d}-{label}-' + ax for ax in 'xyzc')
        for i, label in enumerate(suit.MARKER_LABELS):
            w(headers, i=i, label=label)
        w('\n')

        # write data frames
        for elapsed, prev, curr, frame in self.records:
            w('{},{}', elapsed, effector)
            w(',{t.index},{t.center[0]},{t.center[1]},{t.center[2]}', t=prev)
            w(',{t.index},{t.center[0]},{t.center[1]},{t.center[2]}', t=curr)
            for i in range(len(frame)):
                w(',{m.pos[0]},{m.pos[1]},{m.pos[2]},{m.cond}', m=frame[i])
            w('\n')

        # finish up
        handle.close()
        logging.info('wrote %d records to trial output %s',
                     len(self.records), os.path.basename(output))


class Block(vrlab.Block):
    '''Manage a block of trials in the tracing experiment.

    This class handles block setup (playing a sound, making a directory for
    recording data) and generates trials in the block by choosing sequences of
    targets.
    '''

    def __init__(self, experiment, effector):
        super(Block, self).__init__()

        self.experiment = experiment
        self.effector = effector

        self.trials = targets.CIRCUITS[:6]

        stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.output = os.path.join(
            experiment.output, '{}-block{:02d}'.format(stamp, self.index))

        logging.info('NEW BLOCK -- effector %s trials %s',
                     suit.MARKER_LABELS[self.effector],
                     '|'.join(''.join('{:x}'.format(t) for t in ts)
                              for ts in self.trials))

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
        vrlab.sounds.gong.play()
        self.experiment.prox.clearTargets()

    def generate_trials(self):
        for ts in self.trials:
            yield Trial(self, [targets.NUMBERED[t] for t in ts])


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
        dirname = self.find_output(threshold_min=GAP_MINUTES)
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
        # just run one block at a time to prevent vizard from freezing.
        yield Block(self, effector=suit.MARKERS.R_FING_INDEX)


if __name__ == '__main__':
    logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG,
            format='%(levelname).1s %(asctime)s %(name)s:%(lineno)d %(message)s',
        )
    Experiment().main(fullscreen=False)
