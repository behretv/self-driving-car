""" File to write unit test for the hyper parameter handler """
import pytest
import json

from traffic_sign_detection.hyper_parameter_handler import HyperParameterHandler, Parameters
from traffic_sign_detection.file_handler import FileHandler


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {'epochs': 11, 'batch_size': 129, 'learning_rate': 0.002, 'accuracy': 1.0},
            {'epochs': 11, 'batch_size': 129, 'learning_rate': 0.002, 'accuracy': 0.0}
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
