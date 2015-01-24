import climate
import collections
import datetime
import fnmatch
import functools
import gzip
import hashlib
import io
import itertools
import joblib
import numpy as np
import os
import pandas as pd
import pickle
import re
import scipy.interpolate
import scipy.signal

logging = climate.get_logger('source')


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


def _load_trial(t):
    return t.load()

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

    def trials_matching(self, pattern, load=True):
        matches = []
        for t in self.trials:
            if t.matches(pattern):
                matches.append(t)
        if not load:
            return matches
        pool = joblib.Parallel(-1)
        load = joblib.delayed(_load_trial)
        return pool(load(t) for t in matches)


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
        return (np.asarray(self.df.index[1:]) -
                np.asarray(self.df.index[:-1])).mean()

    @property
    def effector_trajectory(self):
        return self.trajectory(self.lookup_marker(self.df.effector.iloc[0]))

    @property
    def target_trajectory(self):
        return self.trajectory('target')

    @property
    def source_trajectory(self):
        return self.trajectory('source')

    @property
    def marker_channel_columns(self):
        return [c for c in self.df.columns
                if c.startswith('marker') and
                c[6:8].isdigit() and
                c[-1] in 'xyz']

    @property
    def marker_columns(self):
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

    def replace_dropouts(self, threshold=0.1):
        '''For a given marker-start column, replace dropout frames with nans.

        This method alters the movement's `df` in-place.

        Parameters
        ----------
        threshold : float in (0, 1), optional
            Drop an entire marker if fewer than this proportion of frames are
            visible. Defaults to 0.1 (i.e., at least 10% of frames must be
            visible).
        '''
        for marker in self.df.marker_columns:
            start = marker + '-x'
            stop = marker + '-c'
            m = self.df.loc[:, start:stop]
            x, y, z, c = (m[c] for c in m.columns)
            # "good" frames have reasonable condition numbers and are not
            # located *exactly* at the origin (which, for the cube
            # experiment, is on the floor).
            good = (c > 0) & (c < 10) & ((x != 0) | (y != 0) | (z != 0))
            # if fewer than the given threshold of this marker's frames are
            # good, drop the entire marker from the data.
            if good.sum() / len(good) < threshold:
                good[:] = False
            self.df.ix[~good, start:stop] = float('nan')

    def drop_empty_markers(self):
        '''Drop channels from our data frame that do not contain data.
        '''
        empty = []
        for marker in self.marker_columns:
            if self.df[marker + '-c'].count() == 0 and self.df[marker + '-x'].sum() == 0:
                for channel in 'xyzc':
                    empty.append('{}-{}'.format(marker, channel))
        self.df = self.df.drop(empty, axis=1)

    def svt(self, threshold=100, min_rmse=0.01, consec_frames=3, log_every=0):
        '''Complete missing marker data using singular value thresholding.

        This method alters the movement's `df` in-place.

        Singular value thresholding is described in Cai, Candes, & Shen (2010),
        "A Singular Value Thresholding Algorithm for Matrix Completion" (see
        http://arxiv.org/pdf/0810.3286.pdf). The implementation here is rather
        naive but seems to get the job done for the types of mocap data that we
        gathered in the cube experiment.

        Parameters
        ----------
        threshold : int, optional
            Threshold for singular values. Defaults to 100.
        min_rmse : float, optional
            Halt the reconstruction process when reconstructed data is below
            this RMS error compared with measured data. Defaults to 0.01.
        consec_frames : int, optional
            Compute the SVT using trajectories of this many consecutive frames.
            Defaults to 3.
        log_every : int, optional
            Number of SVT iterations between logging output. Defaults to 0,
            which only logs output at the start and finish of the SVT process.
        '''
        cols = [c for c in self.marker_channel_columns if self.df[c].count()]
        odf = self.df[cols]
        num_frames, num_markers = odf.shape
        num_entries = num_frames * num_markers
        filled_ratio = odf.count().sum() / max(1e-3, num_entries)

        # learning rate heuristic, see paper section 5.1.2 for details.
        learning_rate = 1.2 / filled_ratio

        # interpolate linearly and compute weights based on inverse distance to
        # closest non-dropout frame. we want the smoothed data to obey a
        # nearly-linear interpolation close to observed frames, but far from
        # those frames we don't have much prior knowledge.
        linear = odf.interpolate().ffill().bfill()
        weights = pd.DataFrame(np.zeros_like(odf), index=odf.index, columns=odf.columns)
        for marker, columns in itertools.groupby(cols, lambda c: c[:-2]):
            _, closest = self._closest(self.df[marker + '-c'])

            # discount linear interpolation by e^-1 = 0.368 at 200ms,
            # e^-2 = 0.135 at 400ms, etc.
            w = np.exp(-5 * self.approx_delta_t * closest)

            # discount finger markers; the are prone to dropout, and we want
            # their position to influence the overall posture less than markers
            # attached to large limbs.
            if 'fing' in marker:
                w /= 2

            # set weights for x, y, and z channels of this marker
            for column in columns:
                weights[column] = w

        # create dataframe of trajectories by reshaping existing data.
        darr = np.asarray(linear)
        warr = np.asarray(weights)
        extra = num_frames % consec_frames
        if extra > 0:
            z = np.zeros((consec_frames - extra, num_markers))
            darr = np.vstack([darr, z])
            warr = np.vstack([warr, z])
        shape = len(darr) // consec_frames, num_markers * consec_frames
        df = pd.DataFrame(darr.reshape(shape))
        wf = pd.DataFrame(warr.reshape(shape))

        logging.info('SVT: filling %d x %d, reshaped as %d x %d',
                     num_frames, num_markers, df.shape[0], df.shape[1])
        logging.info('SVT: missing %d of %d values (%.1f%% filled)',
                     num_entries - odf.count().sum(),
                     num_entries,
                     100 * filled_ratio)

        def log():
            logging.info('SVT %d: weighted rmse %f using %d singular values',
                         i, rmse, len(s.nonzero()[0]))

        s = None
        x = y = pd.DataFrame(np.zeros_like(df))
        rmse = min_rmse + 1
        i = 0
        while i < 1000 and rmse > min_rmse:
            err = wf * (df - x)
            y += learning_rate * err
            u, s, v = np.linalg.svd(y, full_matrices=False)
            s = np.clip(s - threshold, 0, np.inf)
            x = pd.DataFrame(np.dot(u, np.dot(np.diag(s), v)))
            rmse = np.sqrt((err * err).mean().mean())
            if log_every and i % log_every == 0: log()
            i += 1
        log()

        x = np.asarray(x).reshape((-1, num_markers))[:num_frames]
        df = pd.DataFrame(x, index=odf.index, columns=odf.columns)
        for c in cols:
            self.df[c] = df[c]

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
            x, y, z = np.array([
                np.dot(r, x) for r, x in zip(rots, np.array(self.trajectory(m)))
            ]).T
            self.df[m + '-x'] = x
            self.df[m + '-y'] = y
            self.df[m + '-z'] = z
        self.df['heading'] = angles

    def add_velocities(self):
        '''Add columns to the data that reflect the instantaneous velocity.'''
        dt = 2 * self.approx_delta_t
        for c in self.marker_channel_columns:
            ax = c[-1]
            self.df['{}-v{}'.format(c[:-2], ax)] = pd.rolling_apply(
                self.df[c], 3, lambda x: (x[2] - x[0]) / dt).shift(-1).fillna(0)

    def reindex(self, frame_rate=100.):
        '''Reindex the data frame to a regularly spaced time grid.

        The existing `df` attribute of this Trial will be replaced.

        Parameters
        ----------
        frame_rate : float, optional
            Frame rate for desired time offsets. Defaults to 100Hz.
        '''
        posts = np.arange(0, self.df.index[-1], 1. / frame_rate)
        df = pd.DataFrame(columns=self.df.columns, index=posts)
        for c in self.df.columns:
            series = self.df[c].reindex(posts, method='ffill', limit=1)
            if not c.startswith('marker'):
                series = series.ffill().bfill()
            df[c] = series
        self.df = df
        self._debug('counts after reindexing')

    def lowpass(self, freq=10., order=4, only_dropouts=True):
        '''Filter marker data using a butterworth low-pass filter.

        This method alters the data in `df` in-place.

        Parameters
        ----------
        freq : float, optional
            Use a butterworth filter with this cutoff frequency. Defaults to
            10Hz.
        order : int, optional
            Order of the butterworth filter. Defaults to 4.
        only_dropouts : bool, optional
            Replace only data in dropped frames. Defaults to True. Set this to
            False to replace all data with lowpass-filtered data.
        '''
        nyquist = 1 / (2 * self.approx_delta_t)
        assert 0 < freq < nyquist
        b, a = scipy.signal.butter(order, freq / nyquist)
        for c in self.marker_channel_columns:
            series = self.df[c]
            smooth = scipy.signal.filtfilt(b, a, series)
            if only_dropouts:
                drops = np.asarray(~self.df[c[:-1] + 'c'].notnull())
                series[drops] = smooth[drops]
            else:
                self.df[c] = smooth

    def normalize(self, frame_rate=100., order=1, dropout_decay=0.1, accuracy=1):
        '''Use spline interpolation to resample data on a regular time grid.

        The existing `df` attribute of this Trial will be replaced.

        This method accomplishes two forms of data cleanup at once: filling in
        dropped markers, and aligning the dataset to regularly spaced time
        intervals. We need to accomplish both of these normalization techniques
        together because the alignment process requires fitting a model to the
        data, and the filling-in process similarly requires a model. In
        addition, when performing "pure" alignment, it's hard to say where the
        dropouts occurred after realignment.

        Parameters
        ----------
        frame_rate : float, optional
            Frame rate for desired time offsets. Defaults to 100Hz.

        order : int, optional
            Order of desired interpolation. Defaults to 3 (cubic
            interpolation). Set to 0 for no interpolation.

        dropout_decay : float, optional
            If `order` > 1, perform linear interpolation on each column before
            fitting higher-order splines, and then use the linearly interpolated
            values as guides while fitting the higher-order spline.

            To accomplish this "guiding" process, the values computed by the
            linear interpolation will be assigned weights inversely proportional
            to their distance from the nearest known channel value. This
            parameter gives the slope of the weight decay process, as a
            proportion of overall channel standard deviation. Defaults to 0.3.

        accuracy : float, optional
            Accuracy of the higher-order spline fit relative to observed (and,
            to a lesser extent, linearly-interpolated) data points. Set this to
            a higher value to fit the spline closer to the data, with the
            possible cost of over-fitting. Defaults to 1.
        '''
        # create "fenceposts" for realigning data to regular time intervals.
        dt = 1 / frame_rate
        t0 = self.df.index[0]
        posts = pd.Index(np.arange(dt + t0 - t0 % dt, self.df.index[-1], dt))

        values = []
        for i, column in enumerate(self.df.columns):
            series = self.df[column]

            if series.count() <= order or column.endswith('-c') or not column.startswith('marker'):
                values.append(series.reindex(posts, method='ffill', limit=1))
                logging.debug('%s: reindexed series %d -> %d',
                              column, series.count(), values[-1].count())
                continue

            linear = series.interpolate().fillna(method='ffill').fillna(method='bfill')

            if order == 1:
                values.append(linear.reindex(posts, method='ffill'))
                logging.debug('%s: interpolated series %d -> %d',
                              column, series.count(), values[-1].count())
                continue

            drops, closest = self._closest(series)

            # compute rolling standard deviation of observations.
            w = int(1 / self.approx_delta_t) // 5
            std = pd.rolling_std(linear, w).shift(-w // 2)
            std[std.isnull()] = std.mean()

            # for dropouts, replace standard deviation with a triangular
            # window of uncertainty that increases with distance to the
            # nearest good frame.
            std[drops] = (1 + closest[drops]) * std.mean() * dropout_decay

            # compute higher-order spline fit.
            spl = scipy.interpolate.UnivariateSpline(
                linear.index, linear.values, w=accuracy / std, k=order)

            # evaluate spline at predefined time intervals.
            values.append(spl(posts))
            err = values[-1] - series.reindex(posts, method='ffill')
            logging.debug('%s: interpolated %d points using %d knots: rmse %.3f',
                          column,
                          series.count(),
                          len(spl.get_knots()),
                          np.sqrt((err ** 2).mean()))

            '''
            import lmj.plot
            ax = lmj.plot.axes()
            ax.plot(series.index, series, 'o', alpha=0.3, color=lmj.plot.COLOR11[0])
            ax.plot(linear.index, linear, '+', alpha=0.3, color=lmj.plot.COLOR11[1])
            ax.fill_between(linear.index, linear - std, linear + std, alpha=0.2, lw=0, color=lmj.plot.COLOR11[1])
            ax.plot(spl.get_knots(), spl(spl.get_knots()), 'x', lw=0, mew=2, alpha=0.7, color=lmj.plot.COLOR11[2])
            ax.plot(posts, vals, '-', lw=2, alpha=0.7, color=lmj.plot.COLOR11[2])
            ax.set_xlim(self.df.index[0], self.df.index[-1])
            lmj.plot.show()
            '''

        self.df = pd.DataFrame(dict(zip(self.df.columns, values)))

    def _closest(self, series):
        '''Compute the distance (in frames) to the nearest non-dropout frame.

        Parameters
        ----------
        series : pd.Series
            A Series holding a single channel of mocap data.

        Returns
        -------
        drops : pd.Series
            A boolean series indicating the dropout frames.
        closest : pd.Series
            An integer series containing, for each frame, the number of frames
            to the nearest non-dropout frame.
        '''
        drops = series.isnull()
        closest_l = [0]
        for d in drops:
            closest_l.append(1 + closest_l[-1] if d else 0)
        closest_r = [0]
        for d in drops[::-1]:
            closest_r.append(1 + closest_r[-1] if d else 0)
        return drops, pd.Series(
            list(map(min, closest_l[1:], reversed(closest_r[1:]))),
            index=series.index)

    def drop_fiddly_target_frames(self, enter_threshold=0.25, exit_threshold=0.35):
        '''This code attempts to regularize the moment of target contact.

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
            of 0.2m, so this seems somewhat reasonable.
        exit_threshold : float, optional
            Distance (in meters) from target above which a subject is considered
            to be no longer touching the target. Defaults to 0.35.

        Returns
        -------
        percent_dropped : float
            Percent of frames in this trial that were dropped.

        '''
        dist = self.distance_to_target()
        rate = pd.rolling_mean(dist.diff(2).shift(-1).fillna(0), 4).shift(-2).fillna(0)
        minima = (dist < enter_threshold) & (rate > 0)
        stayings = dist < exit_threshold
        for i in minima[minima].index:
            rest = stayings[i:]
            if sum(rest) > 0.7 * len(rest) and len(rest) > 0.5 / self.approx_delta_t:
                self.df = self.df.drop(rest.index)
                percent_dropped = 100 * len(rest) / len(dist)
                logging.info('dropping %d fiddly frames -- %.1f %% of trial',
                             len(rest), percent_dropped)
                return percent_dropped
        return 0

    def make_body_relative(self):
        '''Translate and rotate marker data so that it's body-relative.
        '''
        r_ilium = self.trajectory('r-ilium')
        l_ilium = self.trajectory('l-ilium')
        r_hip = self.trajectory('r-hip')
        l_hip = self.trajectory('l-hip')
        self.recenter((r_ilium + l_ilium + r_hip + l_hip) / 4)
        r = ((r_hip - r_ilium) + (l_hip - l_ilium)) / 2
        self.rotate_heading(np.arctan2(-r.z, r.x))

    def make_target_relative(self):
        '''Translate and rotate marker data so it's relative to the target.
        '''
        self.recenter(self.target_trajectory)
        r = (self.trajectory('r-ilium') + self.trajectory('l-ilium')) / 2
        self.rotate_heading(np.arctan2(-r.z, r.x))


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

    @property
    def total_distance(self):
        distances = []
        _, (x, y, z) = next(self.source_trajectory.iterrows())
        for _, (u, v, w) in self.target_trajectory.drop_duplicates().iterrows():
            distances.append(np.linalg.norm([x - u, y - v, z - w]))
            x, y, z = u, v, w
        return sum(distances)

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

    def drop_fiddly_target_frames(self, enter_threshold=0.25, exit_threshold=0.35):
        '''Drop fiddly target frames for all target movements in this trial.
        '''
        movements = []
        for t in range(12):
            mov = self.movement_to(t)
            if not 0 < len(mov.df): # < 1000:
                continue
            mov.drop_fiddly_target_frames(enter_threshold, exit_threshold)
            movements.append(mov)
        self.df = pd.concat(movements)

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
