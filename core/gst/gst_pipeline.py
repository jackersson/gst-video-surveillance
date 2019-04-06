from typing import Optional, List

from core.structures import VideoSourceConfig, IModule

from core.structures import SourceElement, SinkElement, TransformElement
from core.structures import QueueBasedElement, LeakyQueueByTopic, BatchElement

from .gst_pipeline_base import GstPipelineBase
from .gst_pipeline_builders import get_gst_pipeline_string_builder
from .gst_buffer_to_frame_data_adapter import GstBufferToFrameDataAdapter


class QueueBasedElementsFactory:

    def __init__(self, maxsize=8, leaky=True):
        self._links = {}

        self._maxsize = maxsize
        self._leaky = leaky

    def from_modules(self, modules: List[IModule], source_id: str) -> List[QueueBasedElement]:
        elements = []
        n = len(modules) - 1
        for i, m in enumerate(modules):
            # current_src = LeakyQueueByTopic()
            previous_src = elements[i - 1].src if i > 0 else None
            if i == 0:
                elements.append(SourceElement(m, src=self._create_link(), topic=source_id))
            elif i == n:
                elements.append(SinkElement(m, topic=source_id, sink=previous_src))
            else:
                if m.is_batch_enabled:
                    batch_element = self._get_batch_element(m)
                    elements[i - 1].src = batch_element.sink
                    elements[i - 1].topic = batch_element.topic
                    elements.append(batch_element)
                else:
                    elements.append(TransformElement(m, src=self._create_link(), sink=previous_src, topic=source_id))

        '''
        # Was just a check that everything connected in a right way
        for i, e in enumerate(elements):
            if i > 0:
                # assert e.sink == elements[i-1].src
                print(f"topics {elements[i-1]}->{e} : ({elements[i-1].topic}) -> ({e.topic})")
        '''
        return elements

    def _get_batch_element(self, module: IModule) -> BatchElement:
        key = hash(module)
        if key not in self._links:
            self._links[key] = BatchElement(module, LeakyQueueByTopic(), LeakyQueueByTopic())
        return self._links[key]

    def _create_link(self):
        return LeakyQueueByTopic(maxsize=self._maxsize, leaky=self._leaky)


class GstPipeline(object):

    _COUNTER = 0
    ELEMENTS_FACTORY = QueueBasedElementsFactory()

    def __init__(self, video_source: VideoSourceConfig, index: Optional[int] = None):
        if index:
            self._idx = index
        else:
            self._idx = self._COUNTER
            self._COUNTER += 1

        # GstBufferToFrameDataAdapter should be first module by default
        video_source.modules = [GstBufferToFrameDataAdapter(source_id=video_source.source_id)] + video_source.modules

        self._elements = self.ELEMENTS_FACTORY.from_modules(video_source.modules, source_id=video_source.source_id)
        # print(f" {index} : {self._elements} ")

        pipeline_builder = get_gst_pipeline_string_builder(video_source, index=index)
        print(f"gst-launch-1.0 {pipeline_builder.gst_launch}")

        pipeline = GstPipelineBase(pipeline_builder.gst_launch)

        for element in self._elements:
            plugin_name = pipeline_builder.module_plugin_name(element.module.module_id())
            plugin = pipeline.get_element(plugin_name)
            if plugin:
                plugin.set_property("model", element)

        plugin = pipeline.get_element(pipeline_builder.fps_plugin_name)
        if plugin:
            plugin.set_property("signal-fps-measurements", True)
            plugin.connect('fps-measurements', self._on_fps)

        self.pipeline = pipeline

    @property
    def elements(self):
        return self._elements

    @property
    def idx(self):
        return self._idx

    def _on_fps(self, element, fps, droprate, avgfps):
        print(f"Pipeline ({self.idx}):  current: {fps} average: {avgfps}")

    @property
    def bus(self):
        return self.pipeline._bus

    def start(self):
        self.pipeline.start()

    def stop(self):
        self.pipeline.stop()
