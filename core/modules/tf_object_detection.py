import os
import yaml
import cv2
import numpy as np
from threading import Lock
from typing import List, Tuple
import tensorflow as tf

from core.structures import FrameData, IModule, ObjectInfo, ImageFormat


def is_gpu(device: str) -> bool:
    return "gpu" in device.lower()


def create_config(device: str = '/device:CPU:0',
                  per_process_gpu_memory_fraction: float = 0.0,
                  log_device_placement: bool = False) -> tf.ConfigProto:

    if is_gpu(device):
        config = tf.ConfigProto(log_device_placement=log_device_placement)
        if per_process_gpu_memory_fraction > 0.0:
            config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        else:
            config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(
            log_device_placement=log_device_placement, device_count={'GPU': 0})

    return config


def parse_graph_def(model_path: str) -> tf.GraphDef:
    model_path = os.path.abspath(model_path)
    assert os.path.isfile(model_path), "Invalid filename {}".format(model_path)
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def import_graph(graph_def: tf.GraphDef, device: str, name: str = "") -> tf.Graph:
    with tf.device(device):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name=name)
            return graph


def load_labels_from_file(filename: str) -> dict:
    assert os.path.isfile(filename), "Invalid filename {}".format(filename)
    labels = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split(":")
            # print(items)
            label_id, label_name = items[:2]
            labels[int(label_id)] = label_name[1:]
    return labels


def load_config(filename: str) -> dict:
    filename = os.path.abspath(filename)
    assert os.path.isfile(filename), "Invalid filename {}".format(filename)

    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream, Loader=yaml.Loader)
            return data
        except yaml.YAMLError as exc:
            raise OSError('Parsing error. Filename: {}'.format(filename))


class TfObjectDetectionModel(object):

    def __init__(self, weights: str,
                 threshold: float = 0.5,
                 device: str = '/device:CPU:0',
                 per_process_gpu_memory_fraction: float = 0.0,
                 log_device_placement=False,
                 labels: List[str] = None,
                 input_shape: Tuple[int, int]=(640, 640)):

        # TODO Docs
        graph_def = parse_graph_def(weights)
        config = create_config(device,
                               log_device_placement=log_device_placement,
                               per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
        graph = import_graph(graph_def, device)

        self.session = tf.Session(graph=graph, config=config)

        # Taken from official website
        self.input = graph.get_tensor_by_name("image_tensor:0")
        self.input_shape = input_shape

        # print([n.name for n in graph.as_graph_def().node][:10])
        # print("Shape : ", self.input.shape)

        # Taken from official website
        output_names = ["detection_classes:0",
                        "detection_boxes:0",
                        "detection_scores:0"]
        self.output = [graph.get_tensor_by_name(name) for name in output_names]

        self.threshold = threshold
        self.labels = labels or {}

        self._box_scaler = None

        self._lock = Lock()

    def process_single(self, image: np.ndarray) -> List[dict]:
        return self._process_safe(np.expand_dims(self._preprocess(image), 0), image.shape[:2][::-1])[0]

    def process_batch(self, images: List[np.ndarray]) -> List[dict]:
        images_ = np.stack([self._preprocess(image) for image in images])
        return self._process_safe(images_, images[0].shape[:2][::-1])

    def _process_safe(self, images: np.ndarray, initial_shape: Tuple[int, int]) -> List[dict]:
        with self._lock:
            return self._process(images, initial_shape)

    def _process(self, images: np.ndarray, initial_shape: Tuple[int, int]) -> List[dict]:
        classes, boxes, scores = self.session.run(self.output,
                                                  feed_dict={self.input: images})

        # _, h, w = images.shape[:3]
        w, h = initial_shape
        box_scaler = np.array([h, w, h, w])

        num_detections = len(classes)
        objects = [[] for _ in range(num_detections)]
        for i in range(num_detections):
            for class_id, box, score in zip(classes[i], boxes[i], scores[i]):

                if class_id not in self.labels or \
                        score < self.threshold:
                    continue

                ymin, xmin, ymax, xmax = list(
                    map(lambda x: int(x), box * box_scaler))
                object_info = {'confidence': float(score),
                               'bounding_box': [xmin, ymin, xmax - xmin, ymax - ymin],
                               'class_name': self.labels[class_id]}

                objects[i].append(object_info)
        return objects

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.input_shape, interpolation=cv2.INTER_NEAREST)

    def __del__(self):
        """ Releases model when object deleted """
        if self.session is not None:
            self.session.close()


def tf_object_detection_model_from_file(filename: str) -> TfObjectDetectionModel:
    """
    :param filename: filename to model config
    """
    return tf_object_detection_model_from_config(load_config(filename))


def tf_object_detection_model_from_config(config: dict) -> TfObjectDetectionModel:
    """
    :param config: model config
    """
    labels = load_labels_from_file(config['labels'])

    return TfObjectDetectionModel(weights=config['weights'],
                                  threshold=config['threshold'],
                                  device=config['device'],
                                  per_process_gpu_memory_fraction=config['per_process_gpu_memory_fraction'],
                                  labels=labels)


class ObjectDetectorAdapter(IModule):
    """ Adapts TfObjectDetectionModel to be injected in Pipeline """

    def __init__(self, model: TfObjectDetectionModel, batch_size: int = 1):
        IModule.__init__(self, batch_size=batch_size)

        self.model = model

        # Warm up model (First launch is too slow)
        self.model.process_single(np.zeros((1, 1, 3), dtype=np.uint8))

    def process(self, data: FrameData, **kwargs) -> FrameData:

        objects = self.model.process_single(data.color)
        data.actives = [ObjectInfo.from_json(obj) for obj in objects]
        return data

    def process_batch(self, data: List[FrameData], **kwargs) -> List[FrameData]:
        images = [frame_data.color for frame_data in data]

        results = self.model.process_batch(images)

        for frame_data, objects in zip(data, results):
            frame_data.actives = [ObjectInfo.from_json(obj) for obj in objects]

        return data

    @property
    def image_format(self):
        return ImageFormat.RGB
