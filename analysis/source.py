import climate
import datetime
import fnmatch
import functools
import numpy as np
import os
import pandas as pd
import scipy.interpolate

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

    def load(self, pattern):
        for s in self.subjects:
            if s.matches(pattern):
                s.load(pattern)


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

    def load(self, pattern):
        for i, b in enumerate(self.blocks):
            if b.matches(pattern):
                b.load(pattern)


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

    def load(self, pattern):
        for t in self.trials:
            if t.matches(pattern):
                t.load()


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

    def load(self, frame_rate=100.):
        import lmj.plot

        self.df = pd.read_csv(self.root, compression='gzip').set_index('time')

        # first replace dropouts with nans.
        def mask_dropouts(col):
            m = self.df.iloc[:, col:col+4]
            x, y, z, c = (m[c] for c in m.columns)
            idx = (c > 0) & (c < 10) & ((x != 0) | (y != 0) | (z != 0))
            self.df.ix[~idx, col:col+4] = float('nan')

        mask_dropouts(9)
        for i, _ in self.markers:
            mask_dropouts(i)

        # for each column, fit a spline and evaluate at regularly spaced times.
        dt = 1 / frame_rate
        start = self.df.index[0]
        posts = np.arange(dt + start - start % dt, self.df.index[-1], dt)
        values = [self.df[c].reindex(posts, method='ffill') for c in self.df.columns[:9]]
        for col in self.df.columns[9:]:
            series = self.df[col].dropna()
            if len(series) > 0 and not col.endswith('-c'):
                values.append(scipy.interpolate.InterpolatedUnivariateSpline(
                    series.index, series)(posts))
            else:
                values.append(self.df[col].reindex(posts, method='ffill'))

        # replace our current DataFrame with the interpolated values.
        self.df = pd.DataFrame(dict(zip(self.df.columns, values)), index=posts)

        logging.info('%s: loaded trial %s', self.basename, self.df.shape)
