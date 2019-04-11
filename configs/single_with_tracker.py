"""
    Runs pipeline for a single video streams with model and tracker
"""

import os

from core.structures import VideoSourceConfig

from core.modules import ObjectDetectionOverlayAdapter, ObjectDetectorAdapter, TrackerIOUAdapter
from core.modules.tf_object_detection import tf_object_detection_model_from_file
from core.modules.object_detection_overlay import ColorPicker, TracksOverlayCairo

colors = ColorPicker()

object_detection_model = tf_object_detection_model_from_file(os.path.abspath("configs/tf_object_api_cfg.yml"))

video_source_config = VideoSourceConfig(
    source="../data/videos/Pyrohova_Street.mp4",
    source_id=0,
    modules=[
        ObjectDetectorAdapter(object_detection_model),
        TrackerIOUAdapter(iou_threshold=0.4, max_frames_count_no_detections=8),
        ObjectDetectionOverlayAdapter(TracksOverlayCairo(colors=colors, line_thickness_scaler=0.0035))
    ],
    show_window=True,
    show_fps=True,
    sync=False
)

VIDEO_SOURCES = [video_source_config]
