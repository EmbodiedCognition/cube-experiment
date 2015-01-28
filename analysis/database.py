import climate
import collections
import datetime
import fnmatch
import functools
import gzip
import hashlib
import io
import numpy as np
import os
import pandas as pd
import pickle
import re

logging = climate.get_logger('database')


def pickled(f, cache='pickles'):
    '''Decorator for caching expensive function results in files on disk.

    This decorator can be used on functions that compute something expensive
    (e.g., loading all of the experiment data and converting it to some scalar
    values for making plots).

    When applied, the wrapped function will be called only if a pickle does not
    exist for a previous call to the function with the same arguments. If a
    pickle does exist, its contents will be loaded and returned instead of
    calling the wrapped function. If a pickle does not exist, the wrapped
    function will be called and a copy of its results will be written to a
    pickle before being returned.

    Pickle files must be removed out-of-band.

    Parameters
    ----------
    f : callable
        A function to wrap.
    cache : str
        Directory location for storing cached results.
    '''
    if not os.path.isdir(cache):
        os.makedirs(cache)
    def wrapper(*args, **kwargs):
        h = hashlib.md5()
        for a in args:
            h.update(str(a).encode('utf8'))
        for k in sorted(kwargs):
            h.update('{}={}'.format(k, kwargs[k]).encode('utf8'))
        tmpfile = os.path.join(cache, '{}-{}.pkl.gz'.format(
            f.__name__, h.hexdigest()))
        if os.path.exists(tmpfile):
            with gzip.open(tmpfile, 'rb') as handle:
                return pickle.load(handle)
        result = f(*args, **kwargs)
        with gzip.open(tmpfile, 'wb') as handle:
            pickle.dump(result, handle)
        return result
    return wrapper


class TimedMixin:
    '''This mixin is for handling filenames that start with timestamps.
    '''

    FORMAT = '%Y%m%d%H%M%S'

    @property
    def stamp(self):
        return datetime.datetime.strptime(
            self.basename.split('-')[0], TimedMixin.FORMAT)

    @property
    def key(self):
        return self.basename.split('.')[0].split('-', 1)[1]


class TreeMixin:
    '''This class helps navigate file trees.

    It's written for data stored as a tree of directories containing data at the
    leaves. Relies on the following attributes being present:

    Attributes
    ----------
    parent : any
        Follow this to navigate up in the tree.
    children : list
        Follow these to navigate down in the tree.
    '''

    @property
    def root(self):
        return os.path.join(self.parent.root, self.basename)

    @functools.lru_cache(maxsize=100)
    def matches(self, pattern):
        return any(c.matches(pattern) for c in self.children)


class Experiment:
    '''Encapsulates all data gathered from the cube poking experiment.

    Attributes
    ----------
    root : str
        The root filesystem path containing experiment data.
    '''

    def __init__(self, root):
        self.root = root
        self.subjects = [Subject(self, f) for f in sorted(os.listdir(root))]
        self.df = None

    @property
    def trials(self):
        for s in self.subjects:
            yield from s.trials

    def trials_matching(self, pattern):
        for t in self.trials:
            if t.matches(pattern):
                yield t


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
        self.blocks = self.children = [
            Block(self, f) for f in sorted(os.listdir(self.root))]
        logging.info('subject %s: %d blocks, %d trials',
                     self.key, len(self.blocks),
                     sum(len(b.trials) for b in self.blocks))
        self.df = None

    @property
    def trials(self):
        for b in self.blocks:
            yield from b.trials


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
        self.trials = self.children = [
            Trial(self, f) for f in sorted(os.listdir(self.root))]

    @property
    def block_no(self):
        return int(re.match(r'-block(\d\d)', self.root).group(1))


class Movement:
    '''Base class for representing and manipulating movement data.

    One movement basically consists of a bunch of motion capture frames sampled
    more or less regularly over a contiguous time interval. Movement data (as in
    the cube experiment) can be augmented with various additional information
    per frame, like the goal of the movement etc.

    This base class contains helper methods for extracting specific subsets of
    information from a movement: trajectories (over time) for a single marker,
    sequences of consecutive frames for some interesting event, etc.
    '''

    def __init__(self, df=None):
        self.df = df

    @property
    def approx_delta_t(self):
        '''Compute the approximate time interval between successive frames.'''
        return pd.Series(self.df.index).diff().mean()

    @property
    def effector_trajectory(self):
        return self.trajectory(self.lookup_marker(self.df.effector.iloc[0]))

    @property
    def target_trajectory(self):
        '''Get the "trajectory" of the target during this movement/trial.'''
        return self.trajectory('target')

    @property
    def source_trajectory(self):
        '''Get the "trajectory" of the source during this movement/trial.'''
        return self.trajectory('source')

    @property
    def marker_channel_columns(self):
        '''Get a list of the x, y, and z columns for marker data.'''
        return [c for c in self.df.columns
                if c.startswith('marker') and
                c[6:8].isdigit() and
                c[-1] in 'xyz']

    @property
    def marker_columns(self):
        '''Get a list of the column prefixes for marker data.'''
        return sorted(set(
            c[:-2] for c in self.df.columns
            if c.startswith('marker') and
            c[6:8].isdigit() and
            c[-2:] == '-c'))

    def lookup_marker(self, marker):
        '''Look up a marker either by index or by name.

        Parameters
        ----------
        marker : str or int
            An integer marker index, or some sort of search string.

        Returns
        -------
        str :
            The name of the matching marker.

        Raises
        ------
        KeyError :
            If there is no matching marker.
        ValueError :
            If there is more than one match for the search string.
        '''
        if not isinstance(marker, str):
            marker = 'marker{:02d}'.format(int(marker))
        matches = [c for c in self.df.columns if marker in c and c.endswith('-x')]
        if len(matches) == 0:
            raise KeyError(marker)
        if len(matches) == 1:
            return matches[0][:-2]
        raise ValueError('more than one match for {}'.format(marker))

    def trajectory(self, marker):
        '''Return the x, y, and z coordinates of a marker in our movement.

        Parameters
        ----------
        marker : str or int
            A search string or integer index to use for looking up the desired
            marker.

        Returns
        -------
        pd.DataFrame :
            A data frame containing x, y, and z columns for each frame in our
            movement.

        Raises
        ------
        KeyError :
            If there is no matching marker.
        ValueError :
            If there is more than one match for the marker search string.
        '''
        marker = self.lookup_marker(marker)
        z = np.asarray(self.df.loc[:, marker + '-x':marker + '-z'])
        return pd.DataFrame(z, index=self.df.index, columns=list('xyz'))

    @property
    def distance_to_target(self):
        '''Return the distance to the target cube over time.

        Returns
        -------
        pd.Series :
            The Euclidean distance from the effector to the target over the
            course of our movement.
        '''
        df = self.effector_trajectory - self.target_trajectory
        return np.sqrt(df.x * df.x + df.y * df.y + df.z * df.z)

    @property
    def distance_from_source(self):
        '''Return the distance to the source cube over time.

        Returns
        -------
        pd.Series :
            The Euclidean distance from the effector to the source over the
            course of our movement.
        '''
        df = self.effector_trajectory - self.source_trajectory
        return np.sqrt(df.x * df.x + df.y * df.y + df.z * df.z)

    def clear(self):
        '''Remove our data frame from memory.'''
        self.df = None

    def mask_dropouts(self):
        '''For each marker, replace dropout frames with nans.

        This method alters the movement's `df` in-place.
        '''
        for marker in self.marker_columns:
            cond = marker + '-c'
            mask = self.df[cond].isnull() | (self.df[cond] < 0) | (self.df[cond] > 100)
            self.df.ix[mask, marker + '-x':cond] = float('nan')

    def recenter(self, center):
        '''Recenter all position data relative to the given centers.

        This method adds three new columns to the data frame called
        center-{x,y,z}.

        Parameters
        ----------
        center : pd.DataFrame
            A data frame containing the center of our x values. This frame must
            have 'x', 'y', and 'z' columns.
        '''
        for c in self.df.columns:
            if c.endswith('-x'): self.df[c] -= center.x
            if c.endswith('-y'): self.df[c] -= center.y
            if c.endswith('-z'): self.df[c] -= center.z
        self.df['center-x'] = center.x
        self.df['center-y'] = center.y
        self.df['center-z'] = center.z

    def rotate_heading(self, angles):
        '''Rotate all marker data using the given sequence of matrices.

        This method adds a new column to the data frame called 'heading'.

        Parameters
        ----------
        angles : sequence of float
            A sequence of rotation angles.
        '''
        def roty(theta):
            ct = np.cos(theta)
            st = np.sin(theta)
            return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        rots = [roty(a) for a in angles]
        for m in self.marker_columns:
            x, _, z = np.array([
                np.dot(r, t) for r, t in zip(rots, self.trajectory(m).values)
            ]).T
            self.df[m + '-x'] = x
            self.df[m + '-z'] = z
        self.df['heading'] = angles

    def add_velocities(self, smooth=19):
        '''Add columns to the data that reflect the instantaneous velocity.'''
        dt = 2 * self.approx_delta_t
        for c in self.marker_channel_columns:
            self.df['{}-v{}'.format(c[:-2], c[-1])] = pd.rolling_mean(
                self.df[c].diff(2).shift(-1) / dt, smooth, center=True)

    def reindex(self, frame_rate=100.):
        '''Reindex the data frame to a regularly spaced time grid.

        The existing `df` attribute of this Trial will be replaced.

        Parameters
        ----------
        frame_rate : float, optional
            Frame rate for desired time offsets. Defaults to 100Hz.
        '''
        posts = np.arange(0, self.df.index[-1], 1. / frame_rate)
        df = self.df.reindex(posts, method='bfill', limit=1)
        for c in df.columns:
            if not c.startswith('marker'):
                df[c] = df[c].bfill().ffill()
        self.df = df
        self._debug('counts after reindexing')

    def mask_fiddly_target_frames(self, enter_threshold=0.25, exit_threshold=0.5):
        '''This code attempts to normalize the moment of target contact.

        Sometimes (and particularly for some targets, like #1) the phasespace
        system was not able to record the relevant marker at the moment of
        crossing the target-is-touched event sphere. In these cases, subjects
        were instructed to sort of wave their hand around the target until the
        system detected the touch event. This process occasionally took many
        seconds, so we'd like to exclude this fiddly data from our movements if
        possible.

        One way to do this would be based on the mean/stdev of the trial
        durations, but I haven't found a strong enough pattern there to be
        reliable. Instead, here we use the smoothed movement data and some
        heuristics about what is likely to constitute a target-touch event to
        exclude these "fiddly" frames from the end of a movement toward a
        particular target.

        Parameters
        ----------
        enter_threshold : float, optional
            Distance (in meters) from target below which a target can be
            considered to be touched. Defaults to a rather large-seeming 25cm,
            but the vizard/phasespace setup was triggered at a nominal distance
            of 20cm, so this seems somewhat reasonable.
        exit_threshold : float, optional
            Distance (in meters) from target above which a subject is considered
            to be no longer touching the target. Defaults to 0.5.
        '''
        changes = self.df.target.diff(1).fillna(0).nonzero()

        dist = self.distance_to_target
        rate = pd.rolling_mean(dist.diff(2).shift(-1), 19, center=True)

        minima = (dist < enter_threshold) & (rate > 0)
        stayings = dist < exit_threshold

        mask = pd.Series([False] * len(minima), index=dist.index)
        for i in minima[minima].index:
            mask = mask

        self.df[mask] = float('nan')
        logging.info('masked %d fiddly frames -- %.1f %% of trial',
                     sum(mask), 100 * sum(mask) / len(self.df))

    def convert_markers_to_z_scores(self):
        '''Convert marker positions to z-scores.'''
        for c in self.marker_channel_columns:
            mean = self.df[c].mean()
            std = max(1e-8, self.df[c].std())
            self.df[c + '-mean'] = mean
            self.df[c + '-std'] = std
            self.df[c] -= mean
            self.df[c] /= std

    def make_body_relative(self):
        '''Translate and rotate marker data so that it's body-relative.'''
        t = self.trajectory
        self.recenter((t('r-hip') + t('r-ilium') + t('l-hip') + t('l-ilium')) / 4)
        r = ((t('r-hip') - t('r-ilium')) + (t('l-hip') - t('l-ilium'))) / 2
        self.rotate_heading(np.arctan2(r.z, r.x))
        self.convert_markers_to_z_scores()

    def make_target_relative(self):
        '''Translate and rotate marker data so it's relative to the target.'''
        self.recenter(self.target_trajectory)
        r = self.source_trajectory
        self.rotate_heading(np.arctan2(r.z, r.x))


class Trial(Movement, TimedMixin, TreeMixin):
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
        super().__init__()
        self.parent = block
        self.basename = basename

    @property
    def trial_no(self):
        return int(re.match(r'-trial(\d\d)-', self.root).group(1))

    @property
    def block_no(self):
        return self.parent.block_no

    def movement_from(self, source):
        '''Return frames that entail movement away from a particular cube.

        Returns
        -------
        Movement :
            A movement object that only include specific frames of data.
        '''
        return Movement(self.df[self.df.source == source])

    def movement_to(self, target):
        '''Return frames that entail movement towards a particular cube.

        Returns
        -------
        Movement :
            A movement object that only include specific frames of data.
        '''
        return Movement(self.df[self.df.target == target])

    def movement_near(self, target, frames=100):
        '''Return frames that entail movement near a particular cube.

        Returns
        -------
        Movement :
            A movement object that only include specific frames of data.
        '''
        mask = (self.df.target == target).shift(frames).fillna(False)
        n = mask.sum()
        if n > 2 * frames:
            # this bizarre expression sets elements of the mask to False,
            # starting from the beginning, so that the total number of True
            # elements in the mask equals 2 * frames. the mask[mask] part gets
            # just the subseries that's True, then mask[foo.index] = False
            # isolates elements in the mask by index and sets them to False.
            mask[mask[mask][:n - 2 * frames].index] = False
        return Movement(self.df[mask])

    @functools.lru_cache(maxsize=100)
    def matches(self, pattern):
        return fnmatch.fnmatch(self.root, pattern)

    def load(self):
        self.df = pd.read_csv(self.root, compression='gzip').set_index('time')
        logging.info('%s %s %s: loaded trial %s',
                     self.parent.parent.key,
                     self.parent.key,
                     self.key,
                     self.df.shape)
        self._debug('loaded data counts')
        return self

    def trial_lengths(self):
        lengths = collections.defaultdict(int)
        for t in self.df.target.unique():
            df = self.movement_to(t).df
            lengths[(df.source.unique()[-1], t)] = len(df)
        logging.info('trial lengths %s', ','.join(str(lengths[(i, j)]) for i in range(12) for j in range(12)))
        return lengths

    def save(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        s = io.StringIO()
        self.df.to_csv(s, index_label='time')
        with gzip.open(path, 'w') as handle:
            handle.write(s.getvalue().encode('utf-8'))
        logging.info('%s %s %s: saved to %s',
                     self.parent.parent.key,
                     self.parent.key,
                     self.key,
                     path)

    def _debug(self, label):
        logging.debug(label)
        for c in self.df.columns:
            logging.debug('%30s: %6d of %6d values',
                          c, self.df[c].count(), len(self.df[c]))
