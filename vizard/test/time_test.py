#!/usr/bin/env python

'''Estimate the minimum delta available using different timers.'''

import time
TICKERS = [time.time, time.clock]
try:
    import viz
    TICKERS.append(viz.tick)
except ImportError:
    pass


def mean(xs):
    return sum(xs) / len(xs)


def sample_sleep():
    t0 = time.clock()
    time.sleep(1e-9) # sleep for 1us
    return time.clock() - t0


def sample_time(tick):
    t0 = t1 = tick()
    while t1 == t0:
        t1 = tick()
    return t1 - t0


if __name__ == '__main__':
    print time.sleep, mean([sample_sleep() for _ in range(20)]), 'sec'
    for tick in TICKERS:
        print tick, mean([sample_time(tick) for _ in range(20)]), 'sec'
