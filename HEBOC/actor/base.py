from abc import ABC


class BaseActor(ABC):
    def __init__(self):
        pass

    def observe(self):
        raise NotImplementedError

    def suggest(self):
        raise NotImplementedError
