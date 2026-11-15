#!/usr/bin/env python3
"""Parser of results/training_log.txt (loss terms of every epoch).
"""
import re, numpy as np
from lib import MAP, R
LOG = {}
cur = None
for l in open(R + 'training_log.txt', encoding='utf8'):
    if l.startswith('=== '):
        cur = l.strip()[4:]
        LOG[cur] = {'ep': [], 't': None}
        continue
    m = re.match('\\s*\\[R=\\s*(\\d+)h\\] ep\\s*(\\d+)/(\\d+) loss=(\\S+) phys=(\\S+) bc=(\\S+) spec=(\\S+) mass=(\\S+) terr=(\\S+) windm=(\\S+) hispec=(\\S+) zm=(\\S+)', l)
    if m:
        LOG[cur]['ep'].append((int(m.group(1)), int(m.group(2)), int(m.group(3))) + tuple((float(x) for x in m.groups()[3:])))
    m = re.match('\\[time\\] ([\\d.]+) s', l)
    if m:
        LOG[cur]['t'] = float(m.group(1))

def get(g, s, R_, ep, col):
    rows = [e for e in LOG[f'{g}_seed{s}']['ep'] if e[0] == R_]
    e = rows[ep]
    return e[3 + col]
