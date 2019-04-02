import numpy as np
from typing import List

from pygst_utils import gst_buffer_to_ndarray

from core.structures import IModule, FrameData


class GstBufferToFrameDataAdapter(IModule):
    """ Converts Gst.Buffer to np.ndarray (RGB-colorspace) """

    def __init__(self, source_id: str, channels: int = 3, batch_size: int = 1):
        """
        :param source_id: unique identifier of video source
        :param channels: num of channels in image colorspace (RGB = 3)
        """

        IModule.__init__(self, batch_size=batch_size)

        self.source_id = source_id
        self._channels = channels
        self._frame_offset = 0

    def process(self, data: FrameData, **kwargs) -> FrameData:
        """
        :param data: None, because module is created for module creation
        """
        buffer = kwargs.pop('buffer')
        width = kwargs.pop("width")
        height = kwargs.pop("height")

        frame = gst_buffer_to_ndarray(buffer, width, height, self._channels)

        result = FrameData(self.source_id, color=frame, frame_offset=self._frame_offset)

        self._frame_offset += 1

        return result

    def process_batch(self, data: List[FrameData], **kwargs) -> List[FrameData]:
        raise NotImplementedError(f"process_batch() {self.__class__.__name__}")
