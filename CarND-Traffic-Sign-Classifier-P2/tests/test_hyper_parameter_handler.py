""" File to write unit tests for the hyper parameter handler """
import pytest
import json

from src.traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler, Parameters
from src.traffic_sign_detection.file_handler import FileHandler


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {"drop_out": 0.5, "out_size": [111, 88], "batch_size": 119, "learning_rate": 0.001, "cnn_filter_size": 5,
             "sigma": 0.1, "epochs": 11, "accuracy": 0.947, "cnn_depth": [6, 16]},
            {"drop_out": 0.5, "out_size": [111, 88], "batch_size": 119, "learning_rate": 0.001, "cnn_filter_size": 5,
             "sigma": 0.1, "epochs": 11, "accuracy": 0.947, "cnn_depth":[6, 16]}
        )
    ]
)
def test_setter_and_getter(test_input, expected):
    files = FileHandler()
    hyper = HyperParameterHandler(files)
    hyper.parameter = Parameters(test_input)
    result = hyper.parameter.to_dict()
    result = json.dumps(result, sort_keys=True)
    expected = json.dumps(expected, sort_keys=True)
    assert result == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([1.0, 1000], True),
        ([0.0, 1000], False),
    ]
)
def test_update(test_input, expected):
    files = FileHandler()
    hyper = HyperParameterHandler(files)
    hyper.update_accuracy(test_input[0], test_input[1])
    result = hyper.is_accuracy_improved
    assert result == expected
