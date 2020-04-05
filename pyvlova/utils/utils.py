import base64
from collections import Mapping
from itertools import chain


def filter_contains(iterable, *containers):
    s = set(chain(*containers))
    if isinstance(iterable, Mapping):
        return {k: v for k, v in iterable.items() if v in s}
    return filter(s.__contains__, iterable)


def slugify(s):
    if s and s[0].isdigit():
        s = '_' + s
    res = []
    for c in s:
        if c.isalnum():
            res.append(c)
        else:
            res.append('_')
    return ''.join(res)
