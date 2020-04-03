from contextlib import contextmanager


class Mode(object):
    def __init__(self, default_mode=None):
        self.default_mode = default_mode
        self.mode = None

    @contextmanager
    def under(self, mode=None):
        old_mode = self.mode
        if mode is None:
            mode = self.default_mode
        self.mode = mode
        assert mode
        yield
        self.mode = old_mode
