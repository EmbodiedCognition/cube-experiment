import climate
import datetime.datetime
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

    def load(self, match_subjects=None, match_blocks=None, match_trials=None, interpolate=True):
        dfs = []
        keys = []
        for s in self.subjects:
            if match_subjects is None or re.search(match_subjects, s.key):
                dfs.append(s.load(match_blocks=match_blocks, match_trials=match_trials, interpolate=interpolate))
                keys.append(s.key)
        self.df = pd.DataFrame(dfs, index=[keys, dfs.index])


class TimedMixin:
    TIMESTAMP_FORMAT = '%Y%m%d%H%M%S'

    @property
    def stamp(self):
        return datetime.datetime.strptime(
            self.basename.split('-')[0], TimedMixin.TIMESTAMP_FORMAT)


class Subject(TimedMixin):
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
        self.experiment = experiment
        self.basename = basename
        self.blocks = [Block(self, f) for f in os.listdir(self.root)]
        logging.info('subject %s: %d blocks', self.key, len(self.blocks))

    @property
    def root(self):
        return os.path.join(self.experiment.root, self.basename)

    @property
    def key(self):
        return self.basename.split('-')[1]

    def load(self, match_blocks=None, match_trials=None, interpolate=True):
        for i, b in enumerate(self.blocks):
            if match_blocks is None or i in match_blocks:
                b.load(match_trials=match_trials, interpolate=interpolate)


class Block(TimedMixin):
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
        self.subject = subject
        self.basename = basename
        self.trials = [Trial(self, f) for f in os.listdir(self.root)]
        logging.info('%s: loaded block with %d trials',
                     self.basename, len(self.trials))

    @property
    def root(self):
        return os.path.join(self.subject.root, self.basename)

    def load(self, match_trials=None, interpolate=True):
        for t in self.trials:
            if match_trials is None or re.search(match_trials, self.basename):
                t.load(interpolate=interpolate)


class Trial(TimedMixin):
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
        self.block = block
        self.basename = basename

    @property
    def markers(self):
        for i, h in enumerate(self.df.columns):
            if h[:2].isdigit() and h.endswith('-x'):
                yield i, h[3:-2]

    def target_contacts(self):
        return self.df['target'].diff().nonzero()[0]

    def marker(self, name):
        i = {h: i for i, h in self.markers}[name]
        return np.asarray(self.df.iloc[:, i:i+3])

    def frame_for_contact(self, target):
        idx = self.df.target.where(target)
        return idx[-1]

    def clear(self):
        self.df = None

    def load(self, interpolate=True):
        path = os.path.join(self.block.root, self.basename)
        self.df = pd.read_csv(path, compression='gzip')
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
