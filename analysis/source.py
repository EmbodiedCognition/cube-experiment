import climate
import datetime
import fnmatch
import functools
import numpy as np
import os
import pandas as pd

logging = climate.get_logger('source')


class Experiment:
    '''Encapsulates all data gathered from the cube poking experiment.

    Attributes
    ----------
    root : str
        The root filesystem path containing experiment data.
    '''

    def __init__(self, root):
        self.root = root
        self.subjects = [Subject(self, f) for f in os.listdir(root)]
        self.df = None

    @property
    def blocks(self, pattern):
        for s in self.subjects:
            yield from s.blocks

    @property
    def trials(self):
        for s in self.subjects:
            yield from s.trials

    def load(self, pattern, interpolate=True):
        for s in self.subjects:
            if s.matches(pattern):
                s.load(pattern, interpolate=interpolate)


class TimedMixin:
    TIMESTAMP_FORMAT = '%Y%m%d%H%M%S'

    @property
    def stamp(self):
        return datetime.datetime.strptime(
            self.basename.split('-')[0], TimedMixin.TIMESTAMP_FORMAT)

    @property
    def key(self):
        return os.path.splitext(self.basename)[0].split('-')[1]


class TreeMixin:
    @property
    def root(self):
        return os.path.join(self.parent.root, self.basename)

    @functools.lru_cache(maxsize=5)
    def matches(self, pattern):
        return any(c.matches(pattern) for c in self.children)


class Subject(TimedMixin, TreeMixin):
    '''Encapsulates data from a single subject.

    Attributes
    ----------
    experiment : `Experiment`
        The experiment object for all subjects.
    blocks : list of `Block`
        A list of the blocks for this subject.
    df : pandas.DataFrame
        A DataFrame object that holds all data for this subject.
    key : str
        A unique string that corresponds to this subject.
    '''

    def __init__(self, experiment, basename):
        self.experiment = self.parent = experiment
        self.basename = basename
        self.blocks = self.children = [Block(self, f) for f in os.listdir(self.root)]
        logging.info('subject %s: %d blocks, %d trials',
                     self.key, len(self.blocks), sum(len(b.trials) for b in self.blocks))
        self.df = None

    @property
    def trials(self):
        for b in self.blocks:
            yield from b.trials

    def load(self, pattern, interpolate=True):
        for i, b in enumerate(self.blocks):
            if b.matches(pattern):
                b.load(pattern, interpolate=interpolate)


class Block(TimedMixin, TreeMixin):
    '''Encapsulates data from a single block (from one subject).

    Parameters
    ----------
    subject : `Subject`
        The subject whose block this is.
    basename : str
        The filesystem directory containing the data for this block.

    Attributes
    ----------
    subject : `Subject`
        The subject whose block this is.
    basename : str
        The filesystem directory containing the data for this block.
    trials : list of `Trial`
        A list of the trials for this subject.
    '''

    def __init__(self, subject, basename):
        self.subject = self.parent = subject
        self.basename = basename
        self.trials = self.children = [Trial(self, f) for f in os.listdir(self.root)]

    def load(self, pattern, interpolate=True):
        for t in self.trials:
            if t.matches(pattern):
                t.load(interpolate=interpolate)


class Trial(TimedMixin, TreeMixin):
    '''Encapsulates data from a single trial (from one block of one subject).

    Parameters
    ----------
    block : `Block`
        The block containing this trial.
    basename : str
        The CSV file containing the data for this trial.

    Attributes
    ----------
    block : `Block`
        The block containing this trial.
    basename : str
        The name of the CSV file containing the trial data.
    df : pandas.DataFrame
        Data for this trial.
    '''

    def __init__(self, block, basename):
        self.block = self.parent = block
        self.basename = basename
        self.df = None

    @property
    def markers(self):
        for i, h in enumerate(self.df.columns):
            if h[:2].isdigit() and h.endswith('-x'):
                yield i, h[3:-2]

    def target_contacts(self):
        return self.df['target'].diff().nonzero()[0]

    def marker_trajectory(self, name):
        i = [i for i, h in self.markers if h == name][0]
        df = self.df.iloc[:, i:i+3].copy()
        df.columns = list('xyz')
        return df

    def frame_for_contact(self, target):
        idx = self.df.target.where(target)
        return idx[-1]

    @functools.lru_cache(maxsize=5)
    def matches(self, pattern):
        return fnmatch.fnmatch(self.root, pattern)

    def clear(self):
        self.df = None

    def load(self, interpolate=True):
        self.df = pd.read_csv(self.root, compression='gzip')
        if interpolate:
            self.interpolate()
        logging.info('%s: loaded trial %s', self.basename, self.df.shape)

    def interpolate(self):
        '''Interpolate missing mocap data.'''
        idx = [i for i, _ in self.markers]
        for c in range(idx[0], idx[-1] + 1, 4):
            markers = self.df.ix[:, c:c+4]
            markers[markers.ix[:, -1] < 0] = float('nan')
            self.df.ix[:, c:c+4] = markers.interpolate()
