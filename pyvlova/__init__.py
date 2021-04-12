# Copyright 2020 Jiang Shenghu
# SPDX-License-Identifier: Apache-2.0

try:
    from . import _ext
except (ImportError, ModuleNotFoundError) as e:
    e.msg += '\nPlease build extensions according to README'
    raise

from . import utils, poly, codegen, op, models
from .poly import trace_mode
from .op import calc_mode
