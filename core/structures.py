import os
import numpy as np
from queue import Queue

from typing import List, Tuple, Optional, Any
from abc import ABCMeta, abstractmethod


class ObjectInfo:
    """ Stores information about object """

    def __init__(self, bounding_box: Tuple[int, int, int, int],
                 confidence: float = 1.0, class_name: str = None):
        """
        :param bounding_box: int[4] - [x, y, w, h]
        :param confidence: [0.0, 1.0]
        """

        self.class_name = class_name
        self.bounding_box = bounding_box
        self.confidence = confidence

    @property
    def as_json(self):
        return {
            "class_name": self.class_name,
            "bounding_box": self.bounding_box,
            "confidence": self.confidence
        }

    @staticmethod
    def from_json(value: dict):
        return ObjectInfo(bounding_box=value["bounding_box"],
                          confidence=value["confidence"],
                          class_name=value["class_name"])


class FrameData:
    """
        Stores information about frame.
        Class is used as unified format to pass from model to model
    """

    def __init__(self, source_id: str,
                 color: np.ndarray,
                 frame_offset: int = 0,
                 actives: List[ObjectInfo] = None):
        """
        :param color: image of size [h, w, c], (RGB-colorspace, c=3)
        :param source_id: unique identifier of video source
        :param offset: frame number from the beginning
        :param objects: list of objects presented on current frame
        """

        self.color = color
        self.source_id = source_id
        self.frame_offset = frame_offset
        self.actives = actives or []

    @property
    def has_color(self) -> bool:
        """ Checks if FrameData contains color image """
        return isinstance(self.color, np.ndarray)


class ImageFormat:
    RGB = 'RGB'
    RGBA = 'RGBA'


class IModule:
    """ Interface for Modules, that could be injected in pipeline """

    __metaclass__ = ABCMeta

    def __init__(self, batch_size: int = 1):
        self._batch_size = batch_size

    @abstractmethod
    def process(self, data: FrameData, **kwargs) -> FrameData:
        """
        :param kwargs: dict of additional arguments
        :type kwargs: any custom parameters
        """

    @abstractmethod
    def process_batch(self, data: List[FrameData], **kwargs) -> List[FrameData]:
        """
        :param kwargs: dict of additional arguments
        :type kwargs: any custom parameters
        """

    @property
    def image_format(self) -> str:
        """Return image color format as string."""
        return ImageFormat.RGB

    @classmethod
    def module_id(cls):
        return cls.__name__

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def is_batch_enabled(self):
        return self._batch_size > 1


class VideoSourceConfig:
    """ Stores information about module """

    def __init__(self, source: str,
                 source_id: str,
                 modules: List[IModule],
                 show_window: bool = False,
                 show_fps: bool = False,
                 sync: bool = False):
        """
        :param source: path to video/camera device/url
        :param source_id: unique video stream identifier
        :param modules: ordered modules in pipeline
        """

        self.source = source
        self.source_id = source_id
        self.modules = modules or []
        self.show_window = show_window
        self.show_fps = show_fps
        self.sync = sync


class QueueBasedElement:

    __metaclass__ = ABCMeta

    def __init__(self, module: IModule, topic: str = "any"):
        self._module = module
        self._topic = topic

    @abstractmethod
    def process(self, timeout=None, **kwargs):
        """
        :param timeout: float (or None) (see Queue.put())
        """
        pass

    @property
    def module(self) -> IModule:
        return self._module

    @property
    def topic(self):
        return self._topic

    @topic.setter
    def topic(self, value):
        self._topic = value


class LeakyQueue(Queue):

    def __init__(self, maxsize: int = 8, leaky: bool = False):
        super().__init__(maxsize=maxsize)
        self._dropped = 0
        self._leaky = leaky

    def put(self, obj, timeout=None):
        if self._leaky and self.full():
            self.get()
            self._dropped += 1
        super().put(obj, timeout=timeout)

    @property
    def dropped(self):
        return self._dropped

    @property
    def size(self):
        return self.qsize()

    def clear(self):
        self._dropped = 0
        while not self.empty():
            self.get()


class LeakyQueueByTopic:

    def __init__(self, maxsize=8, leaky=False):
        self._leaky = leaky
        self._maxsize = maxsize
        self._data = {}

    def add(self, topic: Optional[str] = 'any'):  # queue
        if topic not in self._data:
            self._data[topic] = LeakyQueue(maxsize=self._maxsize, leaky=self._leaky)
        return self._data[topic]

    def put(self, item: Any, topic: str = 'any', timeout=None):
        if topic not in self._data:
            self.add(topic)
        self._data[topic].put(item, timeout=timeout)

    def get(self, topic: str = 'any', timeout=None) -> Any:
        return self._data[topic].get(timeout=timeout) if topic in self._data else None

    @property
    def size(self, topic: str = 'any'):
        return self._data[topic].size

    @property
    def empty(self, topic: str = 'any'):
        return self._data[topic].empty

    def __del__(self):
        for value in self._data.values():
            value.clear()


class SourceElement(QueueBasedElement):
    """ Element wraps IModule to be Source Element:
        - pushes each chunk of data to SRC (DOWNSTREAM)
    """

    def __init__(self, module: IModule, src, topic: str = "any"):
        """
        :param src: queue-based object
        """
        QueueBasedElement.__init__(self, module, topic)
        self._src = src or LeakyQueueByTopic()

    def process(self, timeout=None, **kwargs):
        """ Creates FrameData from IModule and pushes it Forward
        :param kwargs: contain additional information for IModule
        :param timeout: float (or None) (see Queue.put())
        """
        result = self._module.process(None, timeout=timeout, **kwargs)
        self.put(result, timeout=timeout)

    def put(self, data: FrameData, timeout=None):
        """ Push Data Forward """
        self._src.put(data, self.topic, timeout=timeout)

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        self._src = value


class TransformElement(QueueBasedElement):
    """ Element wraps IModule to be Transform Element:
        - accepts data from previous SINK
        - pushes each chunk of data to SRC (DOWNSTREAM)
    """

    def __init__(self, module: IModule, src, sink, topic: str = "any"):
        """
        :param sink: queue based object
        :param src: queue based object
        """
        QueueBasedElement.__init__(self, module, topic)

        self._src = src or LeakyQueueByTopic()
        self._sink = sink or LeakyQueueByTopic()

    def process(self, timeout=None, **kwargs):
        """ Fetches data from SINK/Processes and Pushes it forward to SRC

        :param kwargs: contain additional information for IModule
        :param timeout: float (or None) (see Queue.put())
        """
        data = self.get(timeout=timeout)
        result = data
        if data is not None:
            result = self.module.process(data, timeout=timeout, **kwargs)
        self.put(result, timeout=timeout)

    def get(self, timeout=None) -> FrameData:
        """ Get data from SINK
        :param timeout: float (or None) (see Queue.put())
        """
        return self._sink.get(self.topic, timeout=timeout)

    def put(self, data: FrameData, timeout=None):
        """ Push Data Forward """
        self._src.put(data, self.topic, timeout=timeout)

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        self._src = value

    @property
    def sink(self):
        return self._sink


class SinkElement(QueueBasedElement):
    """ Element wraps IModule to be Sink Element (EndPoint):
        - accepts data from previous SINK (DOWNSTREAM)
    """

    def __init__(self, module: IModule, sink, topic: str = "any"):
        """
        :param sink: queue based object
        """
        QueueBasedElement.__init__(self, module, topic)

        self._sink = sink or LeakyQueueByTopic()

    def process(self, timeout=None, **kwargs) -> FrameData:
        """ Fetch data from SINK
        :param kwargs: contain additional information for IModule
        :param timeout: float (or None) (see Queue.put())
        """
        data = self.get(timeout=timeout)
        result = data
        if data is not None:
            result = self._module.process(data, timeout=timeout, **kwargs)
        return result

    def get(self, timeout=None, **kwargs):
        """ Get data from SINK
        :param timeout: float (or None) (see Queue.put())
        """
        return self._sink.get(self.topic, timeout=timeout)

    @property
    def sink(self):
        return self._sink


class BatchElement(QueueBasedElement):  # OneToManyElement
    """   Element wraps IModule to be Mixer Element (One-To-Many):
        - accepts data from previous SINK
        - pushes each chunk of data to each SRC (DOWNSTREAM)
    """

    def __init__(self, module: IModule, src, sink, topic: str = "any"):
        """
            module: base.IModule
        """

        QueueBasedElement.__init__(self, module, topic)

        self._src = src or LeakyQueueByTopic()
        self._sink = sink or LeakyQueueByTopic()
        self._buffers = []
        self._batch_size = module.batch_size

    def process(self, timeout=None, **kwargs):
        """
            Fetch data from SINK
            Process and Push data Forward (To Src)

            kwargs: dict (could contain any additional information,
                          that could be used in base.IModule
                - timeout: float (or None) (see Queue.put() )
        """
        data = self.get(timeout=timeout)
        if data is None:
            return

        self._buffers.append(data)
        if len(self._buffers) == self.module.batch_size:
            results = self._module.process_batch(self._buffers, timeout=timeout, **kwargs)
            self._buffers = []
            for result in results:
                self.put(result, timeout=timeout)

    def get(self, timeout=None) -> FrameData:
        """ Get data from SINK
        :param timeout: float (or None) (see Queue.put())
        """
        return self._sink.get(self.topic, timeout=timeout)

    def put(self, data: FrameData, timeout=None):
        """ Push Data Forward by Topic """
        self._src.put(data, data.source_id, timeout=timeout)

    @property
    def module(self):
        return self._module

    @property
    def src(self):
        return self._src

    @property
    def sink(self):
        return self._sink


VIDEO_FILES_EXTESIONS = ['.mpg', '.avi', '.mov', '.mp4']


class MediaSourceType:
    FILE = "file"
    WEBCAMERA = "camera"
    RTSP = "rtsp"
    HTTP = "http"


def get_media_source_type(source):
    source = str(source)

    if MediaSourceType.RTSP in source:
        return MediaSourceType.RTSP

    if MediaSourceType.HTTP in source:
        return MediaSourceType.HTTP

    filename = os.path.basename(source)
    _, ext = os.path.splitext(filename)

    if ext.lower() in VIDEO_FILES_EXTESIONS:
        return MediaSourceType.FILE

    return MediaSourceType.WEBCAMERA
