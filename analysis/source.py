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
    def trials(self):
        for s in self.subjects:
            yield from s.trials

    def trials_matching(self, pattern):
        for t in self.trials:
            if t.matches(pattern):
                t.load()
                yield t

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
    def approx_frame_rate(self):
        return (self.df.index[1:] - self.df.index[:-1]).mean()

    @property
    def effector_trajectory(self):
        return self.trajectory('effector')

    @property
    def target_trajectory(self):
        return self.trajectory('target')

    @property
    def source_trajectory(self):
        return self.trajectory('source')

    def trajectory(self, marker):
        x = marker + '-x'
        # if "marker" isn't a column in our dataframe, it might not match
        # exactly because most marker names are prefixed by numbers. so look for
        # a column in the dataframe that ends with the name we want.
        if x not in self.df.columns:
            marker = [c for c in self.df.columns if c.endswith(x)][0][:-2]
        df = self.df.loc[:, marker + '-x':marker + '-z'].copy()
        df.columns = list('xyz')
        return df

    def distance_to_target(self):
        df = self.effector_trajectory - self.target_trajectory
        return np.sqrt(df.x * df.x + df.y * df.y + df.z * df.z)

    def distance_from_source(self):
        df = self.effector_trajectory - self.source_trajectory
        return np.sqrt(df.x * df.x + df.y * df.y + df.z * df.z)

    def clear(self):
        self.df = None

    def _replace_dropouts(self, marker):
        '''For a given marker-start column, replace dropout frames with nans.
        '''
        m = self.df.loc[:, marker + '-x':marker + '-c']
        x, y, z, c = (m[c] for c in m.columns)
        # "good" frames have reasonable condition numbers and are not
        # located *exactly* at the origin (which, for the cube experiment,
        # is on the floor).
        good = (c > 0) & (c < 10) & ((x != 0) | (y != 0) | (z != 0))
        self.df.ix[~good, marker + '-x':marker + '-z'] = float('nan')


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
        self.block = self.parent = block
        self.basename = basename

    def movement_from(self, source):
        return Movement(self.df[self.df.source == source])

    def movement_to(self, target):
        return Movement(self.df[self.df.target == target])

    @functools.lru_cache(maxsize=5)
    def matches(self, pattern):
        return fnmatch.fnmatch(self.root, pattern)

    def load(self):
        self.df = pd.read_csv(self.root, compression='gzip').set_index('time')
        for column in self.df.columns:
            if column.endswith('-c'):
                self._replace_dropouts(column[:-2])
        logging.info('%s: loaded trial %s', self.basename, self.df.shape)

    def svt(self, threshold=700, preserve=0.1):
        '''Complete missing marker data using singular value thresholding.

        Singular value thresholding is described in Cai, Candes, & Shen (2010),
        "A Singular Value Thresholding Algorithm for Matrix Completion" (see
        http://arxiv.org/pdf/0810.3286.pdf). The implementation here is rather
        naive but seems to get the job done for the types of mocap data that we
        gathered in the cube experiment.
        '''
        markers = [
            c for c in self.df.columns
            if c[:2].isdigit() and c[-1] in 'xyz' and self.df[c].count()]

        means = [self.df[c].mean() for c in markers]
        stds = [self.df[c].std() for c in markers]
        zscores = [(self.df[c] - mu) / (std + 1e-10)
                   for c, mu, std in zip(markers, means, stds)]

        df = pd.DataFrame(zscores).T

        n = df.shape[0] * df.shape[1]
        logging.info('SVT: filling data %s, missing %d of %d values',
                     df.shape, n - df.count().sum(), n)

        # learning rate heuristic, see section 5.1.2 for details.
        learning_rate = 1.2 * df.shape[0] * df.shape[1] / df.count().sum()

        def cdf(z): return pd.DataFrame(z, index=df.index, columns=df.columns)

        x = y = cdf(np.zeros_like(df))
        rmse = preserve + 1
        while rmse > preserve:
            err = df - x
            y += learning_rate * err.fillna(0)
            u, s, v = np.linalg.svd(y, full_matrices=False)
            s = np.clip(s - threshold, 0, np.inf)
            rmse = np.sqrt((err * err).mean().mean())
            logging.info('SVT: error %f using %d singular values',
                         rmse, len(s.nonzero()[0]))
            x = cdf(np.dot(u, np.dot(np.diag(s), v)))

        for c, mu, std in zip(markers, means, stds):
            self.df[c] = std * x[c] + mu

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

        start = min(i for i, c in enumerate(self.df.columns) if c.endswith('-c')) - 3
        values = []
        for i, column in enumerate(self.df.columns):
            series = self.df[column]

            if i < start or column.endswith('-c') or series.count() <= order:
                values.append(series.reindex(posts, method='ffill', limit=1))
                logging.info('%s: reindexed series %d -> %d',
                             column, series.count(), values[-1].count())
                continue

            # to start, interpolate observed values linearly.
            linear = series.interpolate().fillna(method='ffill').fillna(method='bfill')

            # compute the distance (in frames) to the nearest non-dropout frame.
            drops = series.isnull()
            closest_l = [0]
            for d in drops:
                closest_l.append(1 + closest_l[-1] if d else 0)
            closest_r = [0]
            for d in drops[::-1]:
                closest_r.append(1 + closest_r[-1] if d else 0)
            closest = pd.Series(
                list(map(min, closest_l[1:], reversed(closest_r[1:]))),
                index=series.index)

            # compute rolling standard deviation of observations.
            w = int(self.approx_frame_rate) // 5
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
            logging.info('%s: interpolated with rmse %.3f',
                         column, np.sqrt((err ** 2).mean()))

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
