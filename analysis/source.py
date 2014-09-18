import climate
import codecs
import csv
import gzip
import numpy as np
import os
import pandas as pd

logging = climate.get_logger('source')


class Subject:
    def __init__(self, root):
        self.root = root
        self.blocks = [Block(os.path.join(root, f)) for f in os.listdir(root)]
        logging.info('%s: loaded subject', root)


class Block:
    def __init__(self, root):
        self.root = root
        self.trials = [Trial(os.path.join(root, f)) for f in os.listdir(root)]
        logging.info('%s: loaded block', os.path.basename(root))


class Trial:
    def __init__(self, filename):
        self.filename = filename
        self.headers = []
        self.df = None

    def clear(self):
        self.df = None

    @property
    def markers(self):
        for i, h in enumerate(self.headers):
            if h[:2].isdigit() and h.endswith('-x'):
                yield i, h[3:-2]

    def load(self):
        self.df = pd.read_csv(self.filename, compression='gzip')
        self.headers = self.df.columns
        logging.info('%s: loaded trial %s', self.filename, self.df.shape)


if __name__ == '__main__':
    climate.enable_default_logging()

    import sys
    s = Subject(sys.argv[1])
    t = s.blocks[0].trials[0]
    t.load()
    for i, h in t.markers:
        print(i, h)
