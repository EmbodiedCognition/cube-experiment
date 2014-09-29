import climate
import datetime
import fnmatch
import functools
import numpy as np
import os
import pandas as pd
import scipy.interpolate
import sklearn.gaussian_process

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

    class ICOL:
        TIME = 0
        SOURCE_ID = 1
        SOURCE_XYZ = [2, 3, 4]
        TARGET_ID = 5
        TARGET_XYZ = [6, 7, 8]
        EFFECTOR_XYZC = [9, 10, 11, 12]

    class COL:
        TIME = 'time'
        SOURCE_ID = 'source'
        SOURCE_XYZ = ['source-x', 'source-y', 'source-z']
        TARGET_ID = 'target'
        TARGET_XYZ = ['target-x', 'target-y', 'target-z']
        EFFECTOR_XYZC = ['effector-x', 'effector-y', 'effector-z', 'effector-c']

    def __init__(self, block, basename):
        self.block = self.parent = block
        self.basename = basename
        self.df = None

    @property
    def marker_columns(self):
        for i, h in enumerate(self.df.columns):
            if h[:2].isdigit() and h.endswith('-x'):
                yield i, h[3:-2]

    @property
    def target_contact_frames(self):
        targets = self.df['target']
        frames, = targets.diff().nonzero()
        return np.concatenate([frames[1:], len(targets)]) - 1

    def marker_trajectory(self, name):
        i = [i for i, h in self.marker_columns if h == name][0]
        df = self.df.iloc[:, i:i+3].copy()
        df.columns = list('xyz')
        return df

    @functools.lru_cache(maxsize=5)
    def matches(self, pattern):
        return fnmatch.fnmatch(self.root, pattern)

    def clear(self):
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.root, compression='gzip').set_index('time')

        # for each marker, replace dropout frames with nans.
        def replace_dropouts(col):
            m = self.df.iloc[:, col:col+4]
            x, y, z, c = (m[c] for c in m.columns)
            # "good" frames have reasonable condition numbers and are not
            # located *exactly* at the origin (which, for the cube experiment,
            # is on the floor).
            good = (c > 0) & (c < 10) & ((x != 0) | (y != 0) | (z != 0))
            self.df.ix[~good, col:col+4] = float('nan')

        replace_dropouts(Trial.ICOL.EFFECTOR_XYZC[0])
        for i, _ in self.marker_columns:
            replace_dropouts(i)

        logging.info('%s: loaded trial %s', self.basename, self.df.shape)

    def realign(self, frame_rate=100., order=1):
        '''Realign raw marker data to regular time intervals.

        If order is nonzero, realignment will also perform spline interpolation
        of the given order.

        The existing `df` attribute of this Trial will be replaced.

        Parameters
        ----------
        frame_rate : float, optional
            Frame rate for desired time offsets. Defaults to 100Hz.
        order : int, optional
            Order of desired interpolation. Defaults to 1 (linear
            interpolation). Set to 0 for no interpolation.
        '''
        dt = 1 / frame_rate
        start = self.df.index[0]
        posts = np.arange(dt + start - start % dt, self.df.index[-1], dt)

        def reindex(column):
            return self.df[column].reindex(posts, method='ffill')

        Spline = scipy.interpolate.InterpolatedUnivariateSpline
        def interp(series):
            return Spline(series.index, series, k=order)(posts)

        def gp(series):
            gp = sklearn.gaussian_process.GaussianProcess(
                nugget=1e-12, theta0=100, #thetaL=10, thetaU=1000,
            )
            gp.fit(np.atleast_2d(series.index).T, series)
            return gp.predict(np.atleast_2d(posts).T)

        MARKERS = Trial.ICOL.EFFECTOR_XYZC[0]
        values = [reindex(c) for c in self.df.columns[:MARKERS]]
        for column in self.df.columns[MARKERS:]:
            series = self.df[column].dropna()
            eligible = len(series) > order and not column.endswith('-c')
            values.append(interp(series) if eligible else reindex(column))

        self.df = pd.DataFrame(dict(zip(self.df.columns, values)), index=posts)
