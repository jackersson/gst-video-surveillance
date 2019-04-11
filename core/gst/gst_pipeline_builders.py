"""
    Generates sequence of plugins for gst-launch-1.0
"""


import os
from typing import Optional, List

from core.structures import VideoSourceConfig
from core.structures import get_media_source_type, MediaSourceType


def gst_named_plugin(plugin_string: str, plugin_name: Optional[str] = None) -> str:
    """ Returns plugin string with name """
    return plugin_string + f" name={plugin_name}" if plugin_name else plugin_string


def gst_synced_plugin(plugin_string: str, sync: bool = False) -> str:
    return plugin_string + f" sync={str(sync)}"


def flatten_list(in_list: List):
    result = []
    for item in in_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def to_gst_string(plugins: List[str]) -> str:
    """ Generates string representation from list of plugins """

    if not isinstance(plugins, list):
        raise ValueError(
            "Invalid type. {} != {}".format(type(plugins), 'list'))

    if len(plugins) <= 0:
        raise ValueError("Empty plugins")

    # plugins_ = []
    plugins_ = flatten_list(plugins)

    result = plugins_[0]
    for i in range(1, len(plugins_)):
        if '.' in plugins_[i][-1]:  # tee case
            result = result + ' ' + plugins_[i]
        else:
            result = result + ' ! ' + plugins_[i]  # ! between plugins
    return result


class GstPipelineStringBuilder:

    def __init__(self, config: VideoSourceConfig, index: int = 0):
        self.index = index
        self.config = config

    @property
    def _gst_source(self) -> str:
        raise NotImplementedError("Not implemented")

    @property
    def source_plugin_name(self) -> str:
        return f"source_{self.index}"

    @property
    def _gst_decoder(self) -> str:
        return "decodebin"

    def module_plugin_name(self, module_id) -> str:
        return f"{module_id}_{self.index}"

    @property
    def fps_plugin_name(self) -> str:
        return f"fps_{self.index}"

    @property
    def _gst_image_sink(self) -> List[str]:
        image_sink = "autovideosink" if self.config.show_window else "fakesink"
        if self.config.show_fps:
            image_sink = f"fpsdisplaysink video-sink={image_sink}"
            image_sink = gst_named_plugin(image_sink, self.fps_plugin_name)
        else:
            image_sink = gst_named_plugin(
                image_sink, self.image_sink_plugin_name)

        image_sink = gst_synced_plugin(image_sink, self.config.sync)
        if "autovideosink" in image_sink:
            image_sink = ['videoconvert', image_sink]
        return image_sink if isinstance(image_sink, list) else [image_sink]

    @property
    def _has_image_sink(self) -> bool:
        return self.config.show_window or self.config.show_fps

    @property
    def image_sink_plugin_name(self) -> str:
        return f"image_sink_{self.index}"

    @property
    def _gst_modules(self) -> List[str]:
        if len(self.config.modules) <= 0:
            return []

        plugins = []
        previous_format = None
        for module in self.config.modules:
            if previous_format != module.image_format:
                plugins.append("videoconvert")
                plugins.append(f"video/x-raw,format={module.image_format}")
                # plugins.append("videoconvert")

            # TODO handle pipeline with not unique names
            plugins.extend([gst_named_plugin("gstplugin_py",
                                             plugin_name=self.module_plugin_name(module.module_id()))])

            previous_format = module.image_format

        return plugins

    @property
    def gst_pipeline(self):
        return [
            self._gst_source,
            self._gst_decoder,
            self._gst_modules,
            self._gst_image_sink
        ]

    @property
    def gst_launch(self):
        return to_gst_string(self.gst_pipeline)


class FileGstPipelineStringBuilder(GstPipelineStringBuilder):

    def __init__(self, config: VideoSourceConfig, index: int = 0):
        GstPipelineStringBuilder.__init__(self, config, index)

    @property
    def _gst_source(self):
        return f"filesrc location={self.config.source}"


def get_gst_pipeline_string_builder(config: VideoSourceConfig, index: int = 0):

    video_source_type = get_media_source_type(config.source)
    if video_source_type == MediaSourceType.FILE:
        return FileGstPipelineStringBuilder(config, index)
    else:
        raise NotImplementedError("Not implemented pipeline builder")
