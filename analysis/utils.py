import climate
import gzip
import hashlib
import os
import pickle

logging = climate.get_logger(__name__)


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
        logging.info('creating pickle cache %s', os.path.abspath(cache))
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
