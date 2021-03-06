import climate
import matplotlib.pyplot as plt
import numpy as np

TARGETS = 12
TRIALS = 6

GOOD_EXAMPLES = [
np.array([[ 7,  3,  8,  0,  5,  1,  6,  2,  9, 11, 10,  4],  # 18
          [ 3,  7,  4, 11,  5,  0, 10,  1,  8,  9,  2,  6],
          [ 8, 11,  0, 10,  2,  9,  6,  1,  7,  4,  5,  3],
          [ 8,  0,  5,  6,  4,  9, 10,  1,  3,  2, 11,  7],
          [ 4,  0,  6, 10,  3,  5,  8, 11,  9,  7,  2,  1],
          [ 5,  9,  0,  1,  4,  3, 11,  2,  8,  6,  7, 10],
          [ 0,  4,  6,  7,  1, 11,  3,  9, 10,  8,  2,  5],
          [ 4,  7,  1,  6, 11,  5,  3,  8, 10,  9,  0,  2],
          [ 8,  3, 11,  1,  9,  6,  5,  2, 10,  0,  4,  7],
          [ 7,  8,  6,  1, 11, 10,  9,  5,  4,  2,  0,  3],
          [ 2,  7, 11,  1,  5,  8,  4,  6,  0,  9,  3, 10],
          [11,  2,  7,  9,  8,  1,  0,  6,  3,  4, 10,  5],
          [ 2,  0,  7,  5, 11,  6,  8,  3,  4,  9,  1, 10],
          [ 5, 10,  6,  0, 11,  4,  3,  1,  2,  8,  7,  9],
          [ 0,  3,  1,  8,  2,  4, 10, 11,  7,  5,  6,  9],
          [10,  5,  7,  6,  9,  4,  2, 11,  3,  0,  8,  1],
          [ 0, 11,  8,  5,  7,  6, 10,  3,  2,  4,  1,  9],
          [11,  4,  8, 10,  7,  0,  1,  5,  9,  2,  3,  6]]),
np.array([[ 3,  6,  7, 10, 11,  9,  5,  2,  0,  1,  8,  4],  # 24
          [ 6, 10,  2, 11,  7,  8,  5,  0,  4,  3,  9,  1],
          [ 6, 11,  7,  8,  1,  2,  4,  0,  3,  5,  9, 10],
          [ 9,  3,  5,  1, 10, 11,  8,  0,  2,  7,  4,  6],
          [ 7, 11,  3,  0,  4,  5,  2, 10,  8,  6,  1,  9],
          [ 9,  7, 10,  0,  2,  5,  8,  3, 11,  6,  4,  1],
          [ 7,  3,  0,  8, 10,  6, 11,  5,  1,  4,  2,  9],
          [ 1,  3,  8,  0,  5,  2,  4, 11, 10,  9,  7,  6],
          [ 2,  0,  5,  7,  9,  6,  8,  3,  1, 11,  4, 10],
          [10,  4,  7,  5,  3,  0,  2,  8,  9, 11,  6,  1],
          [ 8,  3, 10,  7,  2,  1,  9, 11,  5,  4,  0,  6],
          [ 6,  3,  1,  0,  8,  9,  5, 11,  2, 10,  7,  4],
          [ 1,  7,  8, 10,  4,  9, 11,  2,  0,  6,  5,  3],
          [ 7,  1,  4,  6,  8,  2,  3, 11,  9, 10,  5,  0],
          [ 8,  2,  1,  4,  5, 10,  9,  6,  7, 11,  0,  3],
          [ 4,  8, 11,  6,  5,  7,  0,  9,  2, 10,  1,  3],
          [ 0,  7,  4,  1,  6,  2,  9,  8, 11, 10,  3,  5],
          [ 7,  6,  2, 11,  0,  1,  5, 10,  3,  9,  4,  8],
          [ 8,  5,  1,  0, 11,  4, 10,  2,  3,  6,  9,  7],
          [ 4,  9,  2,  6,  3,  7,  0, 11, 10,  5,  8,  1],
          [ 0,  9,  8,  6,  1,  7, 10,  5, 11,  3,  4,  2],
          [ 5,  8,  7,  9,  3,  4,  2, 11,  1,  6,  0, 10],
          [ 6, 10,  1,  9,  0,  3,  2,  7,  5,  4, 11,  8],
          [11,  3, 10,  2,  5,  6,  4,  7,  9,  1,  0,  8]]),
np.array([[ 6, 10,  5,  9,  2,  1,  8, 11,  0,  4,  7,  3],  # 30
          [ 3,  5, 10,  7,  9,  8,  2, 11,  4,  6,  0,  1],
          [ 9,  0,  1,  5,  2,  7, 11,  3,  8,  4, 10,  6],
          [ 3,  4,  9, 11,  2,  6,  0,  7, 10,  5,  8,  1],
          [ 3,  4,  8,  7,  2,  9,  5,  1,  6, 11, 10,  0],
          [ 6,  2,  3, 11,  8, 10,  0,  9,  4,  5,  7,  1],
          [ 0, 10, 11,  2,  1,  3,  5,  4,  9,  8,  7,  6],
          [10,  8,  3, 11,  6,  2,  4,  1,  9,  0,  5,  7],
          [ 3,  7,  0,  4, 11,  5,  2,  8,  6,  9, 10,  1],
          [ 9,  6,  1,  4,  2,  5,  3,  0, 11,  8, 10,  7],
          [ 9, 11,  0,  3,  2,  7, 10,  4,  8,  6,  5,  1],
          [ 8,  5,  7,  6,  4,  2,  9,  3, 10,  0, 11,  1],
          [ 5,  6,  7,  1,  3,  9, 11,  0,  8,  4,  2, 10],
          [ 2,  6,  3,  1,  0,  5,  9,  4, 10, 11,  7,  8],
          [ 7,  5,  8, 11,  1, 10,  3,  2,  4,  0,  9,  6],
          [ 5, 10,  2,  0,  3,  9,  1, 11,  4,  8,  7,  6],
          [ 8,  1,  6, 10,  7,  9,  3,  5,  4,  0,  2, 11],
          [11,  8,  0,  6,  4,  1,  2,  3, 10,  5,  9,  7],
          [ 4,  0,  3,  6,  2, 10,  9,  1, 11,  7,  8,  5],
          [ 6,  9,  1,  7,  4,  5, 11, 10,  8,  3,  0,  2],
          [ 0,  8,  9,  2,  5,  6,  3, 10,  1,  7,  4, 11],
          [ 5,  8,  9, 10,  6,  1,  4,  3,  7,  2,  0, 11],
          [ 7,  9,  5,  4,  1,  2,  0,  8, 11,  3,  6, 10],
          [10,  9,  2,  8,  4,  3,  7,  0,  1,  5, 11,  6],
          [ 7,  8,  5,  1,  4,  3,  0, 10,  6, 11,  2,  9],
          [ 9,  4, 11,  3,  1,  8, 10,  2,  6,  5,  0,  7],
          [ 2,  1,  9,  7,  5,  3,  6,  8,  0, 10, 11,  4],
          [11,  9,  6,  5,  0,  2,  7,  3,  8,  1, 10,  4],
          [ 2,  8,  3, 11,  9,  5, 10,  4,  6,  7,  1,  0],
          [ 1, 10,  3,  4,  7, 11,  5,  9,  0,  6,  8,  2]]),
np.array([[10,  0,  6,  3, 11,  5,  8,  1,  4,  9,  7,  2],  # 36
          [ 6, 10,  4,  2,  9,  0,  8,  7,  5,  1, 11,  3],
          [ 8,  0, 10, 11,  5,  4,  7,  3,  2,  6,  1,  9],
          [ 3,  0,  6,  8,  1,  5,  2,  9, 11,  7,  4, 10],
          [ 5,  7,  9,  3, 10, 11,  2,  1,  4,  8,  6,  0],
          [ 0,  3, 10,  6,  4,  5,  7,  9,  1,  2, 11,  8],
          [ 5,  1, 11,  3,  0,  2,  7, 10,  6,  8,  9,  4],
          [11, 10,  4,  1,  3,  9,  5,  6,  2,  8,  0,  7],
          [ 6,  1,  2,  3,  0,  9,  5, 10,  8,  7, 11,  4],
          [ 9,  4,  8,  7,  2,  0,  1, 10,  5,  3, 11,  6],
          [ 0,  3,  8,  4,  9,  6, 10,  7,  1,  2,  5, 11],
          [ 1, 10,  3,  4,  0,  2,  5,  9,  8, 11,  6,  7],
          [ 7, 11,  8,  3,  6,  5,  2,  1,  0,  4, 10,  9],
          [ 6,  1,  5,  0, 11,  2,  8, 10,  7,  9,  3,  4],
          [ 0, 10,  1,  7,  5,  8, 11,  2,  4,  6,  9,  3],
          [ 9,  1, 10,  3,  7, 11,  8,  0,  5,  4,  6,  2],
          [ 1,  6,  7,  2,  0,  5, 10, 11,  3,  9,  8,  4],
          [11,  9,  1,  7,  0,  4,  3,  2, 10,  5,  8,  6],
          [ 1,  7,  3, 10,  2,  8,  4, 11,  9,  0,  6,  5],
          [ 4, 10,  1,  8,  3,  5,  9,  0, 11,  7,  6,  2],
          [ 7,  8,  2,  0,  9,  4,  1,  3,  5,  6, 11, 10],
          [ 6, 11,  7,  8,  9, 10,  0,  2,  4,  5,  3,  1],
          [ 7,  5,  0, 10,  8,  2, 11,  1,  9,  6,  3,  4],
          [11,  2,  3,  8,  1,  5,  7,  6,  9, 10,  4,  0],
          [ 3,  8,  5, 11, 10,  6,  4,  2,  7,  0,  1,  9],
          [ 5, 11,  6,  8, 10,  0,  1,  3,  7,  4,  9,  2],
          [ 9,  8,  5,  6,  1,  0,  3,  7, 10,  2,  4, 11],
          [ 5,  0,  7,  3,  2, 10,  1,  6,  9, 11,  4,  8],
          [ 3,  9,  7, 10,  8,  6, 11,  0,  5,  4,  2,  1],
          [10,  3,  5,  9,  6,  4,  7,  0,  8,  2, 11,  1],
          [ 9, 11,  0, 10,  4,  8,  5,  2,  7,  6,  3,  1],
          [ 1,  4,  7,  8,  2, 10,  9,  5,  3,  6,  0, 11],
          [ 0,  7,  1,  8, 10,  9,  2,  6,  5,  4,  3, 11],
          [ 0,  4,  6, 10,  2,  9,  7,  8,  3,  1, 11,  5],
          [ 9,  2,  6,  0,  3, 11,  4,  5, 10,  7,  1,  8],
          [ 7,  4, 11,  1,  0,  8,  9, 10,  5,  2,  3,  6]]),
]


def count_transitions(circuits):
    count = np.zeros((TARGETS, TARGETS))
    for circuit in circuits:
        if sum(circuit) == 0:
            break
        for i in zip(circuit[:-1], circuit[1:]):
            count[i] += 1
    return count


def score_circuits(circuits, samples):
    '''Score a set of circuits for overlappiness.

    The score is a combination of the mean overlappiness of random subsets of
    circuits, combined with an exponentially weighted sum of how non-uniform the
    transitions are in the the overall set of circuits.
    '''
    # first measure the amount of expected overlap from random subsets of size
    # TRIALS drawn from this set of circuits. the idea here is to penalize a set
    # of circuits that cannot be sampled randomly with low transition overlap.
    idx = list(range(len(circuits)))
    scores = []
    for _ in range(samples):
        np.random.shuffle(idx)
        c = count_transitions(circuits[idx[:TRIALS]])
        scores.append((c == 0).sum() - TARGETS)
    #print(min(scores), np.mean(scores), max(scores))
    score = np.mean(scores) + 2 * np.std(scores)

    # now measure the overall set of circuits. transitions that are observed
    # only once are penalized a lot, while transitions that happen the expected
    # number of times are penalized just a little.
    count = count_transitions(circuits)
    expect = len(circuits) / TARGETS
    for i in range(int(count.max())):
        score += (2 ** abs(expect - i)) * (count == i).sum()
    return score


def generate_circuits(n, init=None):
    print('Generating {} circuits. Press Ctrl-C to stop.'.format(n))
    idx = list(range(TARGETS))
    circuits = np.zeros((n, TARGETS), int)
    start = 0
    if init is not None:
        start = len(init)
        circuits[:min(n, start)] = init[:min(n, start)]
    for i in range(start, n):
        np.random.shuffle(idx)
        circuits[i] = list(idx)
    best_circuits = None
    best_score = 1e100
    best_time = time = 0
    rng = np.random.randint
    while time - best_time < 1000:
        c = circuits[start + time % (n - start)]
        i, j = rng(TARGETS, size=2)
        while i == j:
            i, j = rng(TARGETS, size=2)
        c[i], c[j] = c[j], c[i]
        try:
            score = score_circuits(circuits, 500)
        except KeyboardInterrupt:
            break
        if score < best_score:
            best_circuits = circuits.copy()
            best_score = score
            best_time = time
            print('{:5d} {:.2f}'.format(time, score))
        else:
            c[i], c[j] = c[j], c[i]
        time += 1
    return best_circuits


def show(c):
    print('score', score_circuits(c, 100))
    print(repr(c))
    t = count_transitions(c)
    im = plt.imshow(t, cmap='hot', origin='lower', interpolation='nearest')
    plt.colorbar(im, ticks=np.linspace(0, t.max(), 1 + t.max()).astype(int))
    plt.show()


STARTER = (
    (10,  0,  1,  3,  8,  4, 11,  7,  9,  6,  5,  2),
    ( 7,  1,  0, 11,  9,  2,  8,  3,  6,  4, 10,  5),
    ( 3,  0,  8, 11,  5, 10,  6,  1,  4,  2,  9,  7),
    (11,  8,  7,  3,  4,  6,  9,  5,  0,  2,  1, 10),
    ( 4,  7,  8,  5,  6,  0,  3,  1,  9, 10,  2, 11),
    (10,  3,  9,  1,  2,  4,  5,  7, 11,  0,  6,  8),
)

def main(n=None):
    if n:
        return show(generate_circuits(int(n), STARTER))
    for t in GOOD_EXAMPLES:
        show(t)


if __name__ == '__main__':
    climate.call(main)
