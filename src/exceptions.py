class NoOtherClusterError(Exception):
    pass


class EmptyClusterError(Exception):
    pass


class SimulationError(Exception):
    def __init__(self, msg, name='', it=0):
        self.msg = msg
        self.name = name
        self.iteration = it
