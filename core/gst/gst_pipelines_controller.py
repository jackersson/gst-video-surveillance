import traceback
from typing import Any

from pygst_utils import Gst, GObject

from .gst_pipeline import GstPipeline


class GstPipelinesController:
    """ Controller for all Gst Pipelines """

    def __init__(self):
        self._main_loop = GObject.MainLoop()
        self._pipelines_by_id = {}
        self._active_pipelines = []

    def append(self, pipeline: GstPipeline):
        """ Appends GstPipeline to list of Pipelines to be run """
        if pipeline.idx not in self._pipelines_by_id:
            self._pipelines_by_id[pipeline.idx] = pipeline
        else:
            raise ValueError(f"Pipeline: {pipeline.idx} - Already exists")

    def run(self):
        pipelines = list(self._pipelines_by_id.values())
        for p in pipelines:
            p.bus.connect("message", self._bus_call, p.idx)
            p.start()

            self._active_pipelines.append(p.idx)

        try:
            self._main_loop.run()
        except:
            traceback.print_exc()
            self.stop()

    def stop(self):
        pipelines = list(self._pipelines_by_id.values())

        for p in pipelines:
            p.stop()

        self.active_pipelines = []
        self._main_loop.quit()

        print(f"All pipelines stopped")

    def _bus_call(self, bus: Gst.Bus, message: Gst.Message, pipeline_id: Any) -> bool:
        if message.type == Gst.MessageType.EOS or \
                message.type == Gst.MessageType.ERROR or \
                message.type == Gst.MessageType.WARNING:

            if pipeline_id in self._pipelines_by_id:
                self._active_pipelines.remove(pipeline_id)
                print(f"Pipeline {pipeline_id} stopped")

            if not self._active_pipelines:
                self.stop()

        return True
