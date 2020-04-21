import json
import os

corpus = []

path = os.path.expanduser('~/data/ptb-dev-diora.parse')

with open(path) as f:
    for line in f:
        ex = json.loads(line)
        y = ex['raw_parse']
        corpus.append(y)

with open('data/dev.txt', 'w') as f:
    for y in corpus:
        f.write('{}\n'.format(y))