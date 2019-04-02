import numpy as np
from unittest import TestCase
import pytest
import cv2

from core.modules import TfObjectDetectionModel
from core.modules.tf_object_detection import tf_object_detection_model_from_file

TF_OBJECT_DETECTION_CONFIG = os.path.abspath("ai-stream/configs/cfg.yml")

IMAGE_SHAPE = (300, 300, 3)
EMPTY_IMAGE = np.full(IMAGE_SHAPE, 255, dtype=np.uint8)

ITERATIONS = 10
ROUNDS = 10


@pytest.fixture
def model():
    model = tf_object_detection_model_from_file(TF_OBJECT_DETECTION_CONFIG)
    model.process_single(EMPTY_IMAGE)
    return model


def test_object_detection_model_tf_single(benchmark, model):
    benchmark.pedantic(model.process_single, args=(EMPTY_IMAGE,), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_double(benchmark, model):
    batch = [EMPTY_IMAGE for _ in range(2)]
    benchmark.pedantic(model.process_batch, args=(batch,), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_triple(benchmark, model):
    batch = [EMPTY_IMAGE for _ in range(3)]
    benchmark.pedantic(model.process_batch, args=(batch,), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_fourth(benchmark, model):
    batch = [EMPTY_IMAGE for _ in range(4)]
    benchmark.pedantic(model.process_batch, args=(batch,), iterations=ITERATIONS, rounds=ROUNDS)


def test_object_detection_model_tf_fifth(benchmark, model):
    batch = [EMPTY_IMAGE for _ in range(5)]
    benchmark.pedantic(model.process_batch, args=(batch,), iterations=ITERATIONS, rounds=ROUNDS)

'''
def test_object_detection_model_tf_fourth_(benchmark, model):
    def process(model, images):
        for image in images:
            _ = model.process_single(image)
    benchmark.pedantic(process, args=(model, [EMPTY_IMAGE for _ in range(4)],), iterations=ITERATIONS, rounds=ROUNDS)
'''