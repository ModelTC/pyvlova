import re

import isl


def get_named_tuples(obj):
    obj = str(obj)
    res = re.findall(r'(\w+)\[(.*?)\]', str(obj))
    for i in range(len(res)):
        res[i] = list(res[i])
        res[i][1] = list(map(str.strip, res[i][1].split(',')))
    return res


def get_unnamed_tuples(obj):
    obj = str(obj)
    res = re.findall(r'\[(.*?)\]', str(obj))
    for i in range(len(res)):
        res[i] = list(map(str.strip, res[i].split(',')))
    return res


def structure_fixed_box(box: isl.fixed_box):
    (name, size), *_ = get_named_tuples(box.size())
    size = list(map(int, size))
    offset = box.offset()
    return name, len(size), size, offset
