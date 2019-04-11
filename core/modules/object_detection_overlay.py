import random
import cairo
import numpy as np
import logging

from typing import Tuple, List, Optional
from pygst_utils import Gst, map_gst_buffer

from core.structures import IModule, FrameData, ImageFormat, ObjectInfo


class ColorPicker:

    def __init__(self):
        self._color_by_id = {}

    def get(self, idx):
        if idx not in self._color_by_id:
            self._color_by_id[idx] = self.generate_color()
        return self._color_by_id[idx]

    def generate_color(self, low=0, high=1):
        return random.uniform(low, high), random.uniform(low, high), random.uniform(low, high)


class ObjectsOverlayCairo:
    """ Draws bounding boxes """

    def __init__(self, line_thickness_scaler: float = 0.0025,
                 font_size_scaler: float = 0.01,
                 font_family: str = 'Sans',
                 font_slant: cairo.FontSlant = cairo.FONT_SLANT_NORMAL,
                 font_weight: cairo.FontWeight = cairo.FONT_WEIGHT_BOLD,
                 text_color: Tuple[int, int, int]=[255, 255, 255],
                 colors: ColorPicker = None):

        self.line_thickness_scaler = line_thickness_scaler
        self.font_size_scaler = font_size_scaler
        self.font_family = font_family
        self.font_slant = font_slant
        self.font_weight = font_weight

        self.text_color = [float(x) / max(text_color) for x in text_color]
        self.colors = colors or ColorPicker()

    def draw(self, buffer: Gst.Buffer, width: int, height: int, objects: List[ObjectInfo]) -> bool:
        try:
            stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, width)
            surface = cairo.ImageSurface.create_for_data(buffer,
                                                         cairo.FORMAT_RGB24,
                                                         width, height,
                                                         stride)
            context = cairo.Context(surface)
        except Exception as e:
            logging.error(e)
            logging.error("Failed to create cairo surface for buffer")
            return False

        try:

            context.select_font_face(self.font_family, self.font_slant, self.font_weight)

            diagonal = (width**2 + height**2)**0.5
            context.set_font_size(int(diagonal * self.font_size_scaler))
            context.set_line_width(int(diagonal * self.line_thickness_scaler))

            for obj in objects:

                r, g, b = self.colors.get(obj.class_name)
                context.set_source_rgb(r, g, b)

                l, t, w, h = obj.bounding_box
                context.rectangle(l, t, w, h)
                context.stroke()

                text = "{}".format(obj.class_name)
                _, _, text_w, text_h, _, _ = context.text_extents(text)

                tableu_height = text_h
                context.rectangle(l, t - tableu_height, w, tableu_height)
                context.fill()

                r, g, b = self.text_color
                context.set_source_rgb(r, g, b)
                context.move_to(l, t)
                context.show_text(text)

        except Exception as e:
            logging.error(e)
            logging.error("Failed cairo render")
            traceback.print_exc()
            return False

        return True


class TracksOverlayCairo(ObjectsOverlayCairo):
    """ Draws bounding boxes, trajectory """

    def __init__(self, line_thickness_scaler: float = 0.0025,
                 font_size_scaler: float = 0.01,
                 font_family: str = 'Sans',
                 font_slant: cairo.FontSlant = cairo.FONT_SLANT_NORMAL,
                 font_weight: cairo.FontWeight = cairo.FONT_WEIGHT_BOLD,
                 text_color: Tuple[int, int, int]=[255, 255, 255],
                 colors: ColorPicker = None,
                 n_trajectory_points=64):

        ObjectsOverlayCairo.__init__(self, line_thickness_scaler=line_thickness_scaler,
                                     font_size_scaler=font_size_scaler,
                                     font_family=font_family,
                                     font_slant=font_slant,
                                     font_weight=font_weight,
                                     text_color=text_color,
                                     colors=colors)

        self._tracks = {}
        self._n_trajectory_points = n_trajectory_points

    def draw(self, buffer: Gst.Buffer, width: int, height: int, objects: List[ObjectInfo]) -> bool:

        for obj in objects:
            if obj.track_id not in self._tracks:
                self._tracks[obj.track_id] = []

            x, y, w, h = obj.bounding_box
            self._tracks[obj.track_id].append([x + w // 2, y + h])

        try:
            stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, width)
            surface = cairo.ImageSurface.create_for_data(buffer,
                                                         cairo.FORMAT_RGB24,
                                                         width, height,
                                                         stride)
            context = cairo.Context(surface)

            context.select_font_face(self.font_family, self.font_slant, self.font_weight)

            diagonal = (width**2 + height**2)**0.5
            context.set_font_size(int(diagonal * self.font_size_scaler))
            line_sickness = int(diagonal * self.line_thickness_scaler)

            for obj in objects:

                track_id = obj.track_id

                r, g, b = self.colors.get(track_id)
                context.set_source_rgb(r, g, b)
                context.set_line_width(line_sickness)

                l, t, w, h = obj.bounding_box
                context.rectangle(l, t, w, h)
                context.stroke()

                text = f"{obj.class_name}"
                _, _, text_w, text_h, _, _ = context.text_extents(text)

                tableu_height = text_h
                context.rectangle(l, t - tableu_height, w, tableu_height)
                context.fill()

                r, g, b = self.text_color
                context.set_source_rgb(r, g, b)
                context.move_to(l, t)
                context.show_text(text)

                context.stroke()

                n = len(self._tracks[track_id]) - 1
                n_from = 0 if n <= self._n_trajectory_points else n - self._n_trajectory_points

                r, g, b = self.colors.get(track_id)
                thickness_scaler = line_sickness ** 3 / self._n_trajectory_points if self._n_trajectory_points > 0 else 1
                self.draw_trajectory_tail(context, list(reversed(self._tracks[track_id][n_from:n])),
                                          buffer=self._n_trajectory_points, color=[r, g, b],
                                          thickness_scaler=thickness_scaler)

        except Exception as e:
            logging.error(e)
            logging.error("Failed cairo render")
            traceback.print_exc()
            return False

    def draw_trajectory_tail(self, context, points, buffer=128, color=[0, 0, 255], thickness_scaler=7.5):
        if len(points) <= 0:
            return

        r, g, b = color
        context.set_source_rgb(r, g, b)

        for i in range(1, len(points)):
            x_from, y_from = points[i - 1]
            x_to, y_to = points[i]

            step = 1 if (buffer - i) == 0 else abs(buffer - i)
            thickness = max(1, int(np.sqrt(step / buffer) * thickness_scaler))

            context.set_line_width(thickness)
            context.move_to(x_from, y_from)
            context.line_to(x_to, y_to)
            context.stroke()


class ObjectDetectionOverlayAdapter(IModule):
    """ Adapts ObjectsOverlayCairo to be injected in Pipeline """

    def __init__(self, model: Optional[ObjectsOverlayCairo] = None, colors: ColorPicker = None, batch_size: int = 1):
        IModule.__init__(self, batch_size=batch_size)

        self.model = model or ObjectsOverlayCairo(colors=colors)

    def process(self, data: FrameData, **kwargs) -> FrameData:

        buffer = kwargs.pop('buffer')
        width = kwargs.pop("width")
        height = kwargs.pop("height")

        with map_gst_buffer(buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as mapped:
            self.model.draw(mapped, width, height, data.actives)

        return data

    def process_batch(self, data: List[FrameData], **kwargs) -> List[FrameData]:
        raise NotImplementedError(f"process_batch() {self.__class__.__name__}")

    @property
    def image_format(self):
        return ImageFormat.RGBA
