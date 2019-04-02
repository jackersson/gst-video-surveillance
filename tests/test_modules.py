import numpy as np
from unittest import TestCase
import pytest
import cv2

from core.modules import TfObjectDetectionModel
from core.modules.tf_object_detection import tf_object_detection_model_from_file

TF_OBJECT_DETECTION_CONFIG = "/home/taras/coder/projects/production-ai/ai-stream/configs/cfg.yml"
BATCH_SIZE = 3

IMAGE_SHAPE = (300, 300, 3)
EMPTY_IMAGE = np.full(IMAGE_SHAPE, 255, dtype=np.uint8)

ITERATIONS = 10
ROUNDS = 10

'''
def test_object_detection_model_tf():
    model = tf_object_detection_model_from_file(TF_OBJECT_DETECTION_CONFIG)
    assert isinstance(model, TfObjectDetectionModel)

    result = model.process_single(EMPTY_IMAGE)
    assert isinstance(result, list)

    result = model.process_batch([EMPTY_IMAGE for _ in range(BATCH_SIZE)])
    assert isinstance(result, list) and len(result) == BATCH_SIZE
'''

@pytest.fixture
def model():
    model = tf_object_detection_model_from_file(TF_OBJECT_DETECTION_CONFIG)
    model.process_single(EMPTY_IMAGE)
    return model


def test_object_detection_model_tf_single(benchmark, model):
    benchmark.pedantic(model.process_single, args=(EMPTY_IMAGE,), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_double(benchmark, model):
    benchmark.pedantic(model.process_batch, args=([EMPTY_IMAGE for _ in range(2)],), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_triple(benchmark, model):
    benchmark.pedantic(model.process_batch, args=([EMPTY_IMAGE for _ in range(3)],), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_fourth(benchmark, model):
    benchmark.pedantic(model.process_batch, args=([EMPTY_IMAGE for _ in range(4)],), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_fifth(benchmark, model):
    benchmark.pedantic(model.process_batch, args=([EMPTY_IMAGE for _ in range(5)],), iterations=ITERATIONS, rounds=ROUNDS)

'''
def test_object_detection_model_tf_fourth_(benchmark, model):
    def process(model, images):
        for image in images:
            _ = model.process_single(image)
    benchmark.pedantic(process, args=(model, [EMPTY_IMAGE for _ in range(4)],), iterations=ITERATIONS, rounds=ROUNDS)
'''