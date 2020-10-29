""" Class to handle files """


class FileHandler:
    def __init__(self):
        self.model_session_file = 'data/parameters/train_model.ckpt'
        self.training_file = 'data/input/train.p'
        self.validation_file = 'data/input/test.p'
        self.testing_file = 'data/input/valid.p'
