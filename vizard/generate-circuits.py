import climate
import matplotlib.pyplot as plt
import numpy as np

TARGETS = 12
TRIALS = 6

GOOD_EXAMPLES = [
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
np.array([[ 7,  3,  5,  2,  8,  1,  0, 11,  6, 10,  4,  9],  # 30
          [ 9,  1,  6,  3,  4,  8, 11,  7, 10,  5,  2,  0],
          [ 2,  5,  3,  4, 10,  0,  1,  8,  7, 11,  6,  9],
          [ 5, 11,  9,  4,  8,  3,  1, 10,  6,  0,  2,  7],
          [ 9,  2,  1, 10,  8,  6,  4,  0,  7,  5, 11,  3],
          [ 7, 10,  3,  4, 11,  5,  9,  0,  1,  2,  6,  8],
          [10, 11,  0,  6,  4,  9,  5,  8,  3,  1,  7,  2],
          [ 4,  7,  9,  1,  6,  0,  2,  5, 11,  8, 10,  3],
          [ 1,  0,  7, 10,  4,  3,  8,  2,  6,  5,  9, 11],
          [ 0,  5,  6,  2,  8,  9,  3,  7,  4, 10, 11,  1],
          [ 9,  6,  1, 11,  4,  3,  8,  2,  7,  0,  5, 10],
          [ 2,  9,  3,  7,  1,  5,  8, 10,  0,  4,  6, 11],
          [ 5,  3,  6, 10,  7,  0, 11,  4,  1,  8,  2,  9],
          [10,  2,  8, 11,  1,  6,  7,  9,  4,  5,  3,  0],
          [ 5,  3,  2,  4,  0,  9,  1, 10,  6, 11,  8,  7],
          [ 7,  4, 11,  5,  1,  8,  6,  2,  3,  9, 10,  0],
          [ 8,  0,  3,  1,  4, 11, 10,  5,  6,  9,  7,  2],
          [ 5,  0,  9,  8, 11,  7,  1,  6,  4, 10,  2,  3],
          [ 5, 10,  1,  3,  0,  9,  7,  8,  4,  6, 11,  2],
          [ 0, 10,  8,  9,  5,  1,  4,  3,  6,  7,  2, 11],
          [ 1,  7,  2, 10,  9,  0,  6,  4, 11,  3,  8,  5],
          [ 5,  7, 11,  2,  4,  1,  9, 10,  6,  0,  8,  3],
          [10,  3,  5,  6,  1, 11,  9,  0,  8,  4,  2,  7],
          [ 9,  8,  4,  0,  2, 11,  6,  3, 10,  5,  7,  1],
          [ 6,  3,  9, 11,  0,  7,  8,  1,  5,  4,  2, 10],
          [ 7,  5,  2,  0,  4,  8,  1,  9,  6,  3, 11, 10],
          [ 7,  6,  5,  0, 10,  3, 11,  9,  2,  1,  8,  4],
          [ 1,  2,  9,  4,  5, 11,  8,  0, 10,  7,  3,  6],
          [11,  4,  7,  6,  0, 10,  1,  3,  2,  5,  8,  9],
          [ 7,  2,  1,  0,  3, 10,  9,  6,  8,  5,  4, 11]]),
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
    idx = list(range(len(circuits)))
    scores = []
    for _ in range(samples):
        np.random.shuffle(idx)
        c = count_transitions(circuits[idx[:TRIALS]])
        scores.append((c == 0).sum() - TARGETS)
    #print(min(scores), np.mean(scores), max(scores))
    score = np.mean(scores) + 2 * np.std(scores)
    count = count_transitions(circuits)
    weight = 1
    for i in range(len(circuits) // TARGETS, -1, -1):
        score += weight * (count == i).sum()
        weight *= 2
    return score


def generate_circuits(n, init=None):
    print('Generating {} circuits. Press Ctrl-C to stop.'.format(n))
    idx = list(range(TARGETS))
    circuits = np.zeros((n, TARGETS), int)
    s = 0
    if init is not None:
        s = len(init)
        m = min(n, s)
        circuits[:m] = init[:m]
    for i in range(s, n):
        np.random.shuffle(idx)
        circuits[i] = list(idx)
    best = None
    scor = 1e100
    t = when = 0
    rng = np.random.randint
    while t - when < 1000:
        c = circuits[t % n]
        i, j = rng(TARGETS, size=2)
        while i == j:
            i, j = rng(TARGETS, size=2)
        c[i], c[j] = c[j], c[i]
        try:
            s = score_circuits(circuits, 500)
        except KeyboardInterrupt:
            break
        if s < scor:
            best = circuits.copy()
            scor = s
            when = t
            print('{:5d} {:.2f}'.format(t, scor))
        else:
            c[i], c[j] = c[j], c[i]
        t += 1
    return best


def show(c):
    print('score', score_circuits(c, 100))
    print(repr(c))
    t = count_transitions(c)
    im = plt.imshow(t, cmap='hot', origin='lower', interpolation='nearest')
    plt.colorbar(im, ticks=np.linspace(0, t.max(), 1 + t.max()).astype(int))
    plt.show()


def main(n):
    show(generate_circuits(int(n)))


if __name__ == '__main__':
    climate.call(main)
