import tvm.autotvm.feature as feature
from tvm.tir import LoweredFunc


def patch_ana_lower():
    if hasattr(feature.ana_lower, 'patched'):
        return
    old_ana_lower = feature.ana_lower
    def ana_lower(s, *args, **kwargs):
        if isinstance(s, LoweredFunc):
            return s
        return old_ana_lower(s, *args, **kwargs)
    feature.ana_lower = ana_lower
    feature.ana_lower.patched = old_ana_lower


def depatch_ana_lower():
    if hasattr(feature.ana_lower, 'patched'):
        feature.ana_lower = feature.ana_lower.patched
