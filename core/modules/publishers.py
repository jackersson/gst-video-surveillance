import os
import json
import base64

from core.structures import FrameData, IModule


class FilePublisher(IModule):
    """ Writes Objects to File in Json Format """

    def __init__(self, filename: str):
        super(FilePublisher, self).__init__()

        self.filename = filename

    def process(self, data: FrameData, **kwargs) -> FrameData:
        with open(self.filename, 'a') as f:

            for active in data.actives:

                message = active.as_json
                message.update({
                    "source_id": data.source_id,
                    "frame_offset": data.frame_offset
                })

                json.dump(message, f)
                f.write('\n')
        return data
