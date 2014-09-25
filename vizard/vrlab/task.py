import logging
import sys

import viz
import vizact
import viztask


class Task(viz.EventClass):
    '''Abstract base class for the workflow during an experiment.

    Instances of class are intended to be used as part of the viztask module, by
    calling `viztask.schedule()` using our `workflow()` method:

    >>> viztask.schedule(MyTaskSubclass().workflow)

    But since this base class is abstract, it is not to be used
    directly---instead, use the Trial, Block, or Experiment classes, or
    implement your own subclass.
    '''

    PRE_RUN_EVENT = viz.getEventID('TASK_PRE_RUN_EVENT')
    POST_RUN_EVENT = viz.getEventID('TASK_POST_RUN_EVENT')

    def __init__(self, periodic=()):
        super(Task, self).__init__()

        self._timers = [0]
        for period, callback in periodic:
            self.add_periodic(period, callback)

    def workflow(self):
        '''Run the workflow associated with this task.

        Each task is separated into three phases: startup, running, and
        teardown. (These are inspired by the commonly implemented phases of a
        unit test.)

        Instances of this task also emit events just before and just after
        running the task. These events can be used to attach callbacks that do
        things like start and stop a timer, or other similar things you might
        want to do during an experiment.
        '''
        yield self.setup()
        viz.sendEvent(self.PRE_RUN_EVENT, self)
        yield self.run()
        viz.sendEvent(self.POST_RUN_EVENT, self)
        yield self.teardown()

    def setup(self):
        '''Set up some environment necessary for running this task.

        This can be used, for example, to:

        - set up storage for variables that might need to be recorded while
          running a task
        - draw or show objects that will be needed for the task
        - etc.

        Example implementations:

            def setup(self):
                os.makedirs(self.trial_output_dir())
                yield viztask.waitTime(3)  # wait 3 seconds

            def setup(self):
                yield viztask.waitKeyDown(' ')  # wait until space is pressed

            def setup(self):
                yield self.signal.wait()  # wait until some signal fires
        '''
        pass

    def run(self):
        '''Run this task.

        Subclasses of this base must implement this method.
        '''
        raise NotImplementedError

    def teardown(self):
        '''Tear down the environment after having run a task.'''
        pass

    def add_periodic(self, period, callback):
        '''Call some code periodically while running a task.

        The given `callback` will be called every `period` seconds between the
        firings of the PRE_RUN and POST_RUN events for this task.

        Parameters
        ----------
        period : float
            Number of seconds between successive calls to the `callback`.
        callback : callable (no arguments)
            Call this function every `period` seconds.

        '''
        timer_id = 1 + max(self._timers)
        self._timers.append(timer_id)

        viz.callback(self.POST_RUN_EVENT, lambda me: viz.killtimer(timer_id))
        def start(me):
            def clock(t):
                if t == timer_id: callback()
            viz.callback(viz.TIMER_EVENT, clock)
            viz.starttimer(timer_id, period, viz.FOREVER)
        viz.callback(self.PRE_RUN_EVENT, start)


class Trial(Task):
    '''A trial represents a single "measurement" during an experiment.

    Trials are grouped into blocks. You should subclass this class and then, at
    a minimum, provide an implementation of the `run()` method to run a single
    trial.

        class MyTrial(Trial):
            def run(self):
                print 'press the space key'
                start = viz.tick()
                yield viztask.waitKeyDown(' ')
                ms = 1000 * (viz.tick() - start)
                print 'reaction time: {:.1f}ms'.format(ms)

    You can set up the environment for a trial by implementing the `setup()`
    method, and clean up or otherwise handle the end of a trial by implementing
    the `teardown()` method.
    '''

    PRE_RUN_EVENT = viz.getEventID('TRIAL_PRE_RUN_EVENT')
    POST_RUN_EVENT = viz.getEventID('TRIAL_POST_RUN_EVENT')


class Block(Task):
    '''A block represents a group of trials during an experiment.

    Blocks are grouped into an experiment. You should subclass this class and
    then, at a minimum, provide an implementation of the `generate_trials()`
    method to create the sequence of trials in this block.
    '''

    PRE_RUN_EVENT = viz.getEventID('BLOCK_PRE_RUN_EVENT')
    POST_RUN_EVENT = viz.getEventID('BLOCK_POST_RUN_EVENT')

    def run(self):
        for trial in self.generate_trials():
            yield trial.workflow()

    def generate_trials(self):
        raise NotImplementedError


class Experiment(Task):
    '''An experiment represents a group of blocks.

    You should subclass this class and then, at a minimum, provide an
    implementation of the `generate_blocks()` method to create the sequence of
    blocks in this experiment.
    '''

    PRE_RUN_EVENT = viz.getEventID('EXPERIMENT_PRE_RUN_EVENT')
    POST_RUN_EVENT = viz.getEventID('EXPERIMENT_POST_RUN_EVENT')

    def run(self):
        for block in self.generate_blocks():
            yield block.workflow()

    def generate_blocks(self):
        raise NotImplementedError

    def main(self, fullscreen=True):
        logging.info('running experiment')
        viz.go(viz.FULLSCREEN if fullscreen else 0)
        viztask.schedule(self.workflow)
