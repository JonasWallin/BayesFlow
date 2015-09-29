class NoDataError(Exception):
    pass


class OldFileError(Exception):
    pass


class NoOtherClusterError(Exception):
    pass


class EmptyClusterError(Exception):
    pass


class SimulationError(Exception):
    def __init__(self, msg, name='', it=0):
        self.msg = msg
        self.name = name
        self.iteration = it


class BadQualityError(Exception):
    pass
