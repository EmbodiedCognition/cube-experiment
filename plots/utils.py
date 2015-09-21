import numpy as np
import matplotlib.colors


COLORS = {
    'marker00-r-head-back': '#9467bd',
    'marker01-r-head-front': '#9467bd',
    'marker02-l-head-front': '#9467bd',
    'marker03-l-head-back': '#9467bd',
    'marker07-r-shoulder': '#111111',
    'marker13-r-fing-index': '#111111',
    'marker14-r-mc-outer': '#111111',
    'marker19-l-shoulder': '#111111',
    'marker25-l-fing-index': '#111111',
    'marker26-l-mc-outer': '#111111',
    'marker31-sternum': '#111111',
    'marker34-l-ilium': '#2ca02c',
    'marker35-r-ilium': '#2ca02c',
    'marker36-r-hip': '#2ca02c',
    'marker40-r-heel': '#1f77b4',
    'marker41-r-mt-outer': '#1f77b4',
    'marker42-r-mt-inner': '#1f77b4',
    'marker43-l-hip': '#2ca02c',
    'marker47-l-heel': '#d62728',
    'marker48-l-mt-outer': '#d62728',
    'marker49-l-mt-inner': '#d62728',
}


RCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.8, 0.8), (1, 0.8, 0.8)),
    green=((0, 0.1, 0.1), (1, 0.1, 0.1)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


GCM = matplotlib.colors.LinearSegmentedColormap('b', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.6, 0.6), (1, 0.6, 0.6)),
    blue= ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


BCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.1, 0.1), (1, 0.1, 0.1)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


OCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 1.0, 1.0), (1, 1.0, 1.0)),
    green=((0, 0.5, 0.5), (1, 0.5, 0.5)),
    blue= ((0, 0.0, 0.0), (1, 0.0, 0.0)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


PCM = matplotlib.colors.LinearSegmentedColormap('r', dict(
    red=  ((0, 0.6, 0.6), (1, 0.6, 0.6)),
    green=((0, 0.4, 0.4), (1, 0.4, 0.4)),
    blue= ((0, 0.7, 0.7), (1, 0.7, 0.7)),
    alpha=((0, 1.0, 1.0), (1, 0.0, 0.0)),
))


# fewf, http://stackoverflow.com/questions/4494404
def contig(cond):
    '''Return (start, end) indices for contiguous blocks where cond is True.'''
    idx = np.diff(cond).nonzero()[0] + 1
    if cond[0]:
        idx = np.r_[0, idx]
    if cond[-1]:
        idx = np.r_[idx, cond.size]
    return idx.reshape((-1, 2))


def local_minima(a):
    '''Return indexes of local minima in a.'''
    minima = numpy.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True]
    return minima.nonzero()[0]
