""" Class to collect hyper parameters and optimize them """


class HyperParameterHandler:
    def __init__(self):
        self._learning_rate = 0.0001
        self._epochs = 10
        self._batch_size = 128

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def epochs(self):
        return self._epochs

    @property
    def batch_size(self):
        return self._batch_size
