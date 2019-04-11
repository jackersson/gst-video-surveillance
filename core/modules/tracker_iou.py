from uuid import uuid4
import traceback
import numpy as np

from typing import List, Optional, Tuple

from core.structures import ObjectInfo, FrameData, IModule


class TrackedObject:

    def __init__(self, idx: int, frame_id: int, object_info: ObjectInfo):
        self._idx = idx
        self._object_info = None
        self._last_frame_update = None

        self.update_from_detection(frame_id, object_info)

    def update_from_detection(self, frame_id: int, object_info: ObjectInfo):
        """ Update with new position from detection """
        object_info.track_id = self._idx
        self._object_info = object_info
        self._last_frame_update = frame_id

    @property
    def object_info(self) -> ObjectInfo:
        return self._object_info

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def bounding_box(self):
        return self._object_info.bounding_box

    @property
    def last_frame_update(self):
        return self._last_frame_update


def intersection_over_union(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """ Calculates intersection over union for two bounding boxes:
        :param boxA: x, y, w, h
        :param boxB: x, y, w, h
    """

    # convert to left, top, right, bottom way
    left_a, top_a, width_a, height_a = boxA
    right_a, bottom_a = left_a + width_a, top_a + height_a

    left_b, top_b, width_b, height_b = boxB
    right_b, bottom_b = left_b + width_b, top_b + height_b

    # determine the (x, y)-coordinates of the intersection rectangle
    left_intersection = max(left_a, left_b)
    top_intersection = max(top_a, top_b)
    right_intersection = min(right_a, right_b)
    bottom_intersection = min(bottom_a, bottom_b)

    if (right_intersection < left_intersection) or (bottom_intersection < top_intersection):
        return 0

    intersection_area = (right_intersection - left_intersection) * (bottom_intersection - top_intersection)

    box_a_area = (right_a - left_a) * (bottom_a - top_a)
    box_b_area = (right_b - left_b) * (bottom_b - top_b)

    iou = intersection_area / float(box_a_area + box_b_area - intersection_area)

    return iou


class IouObjectTracker(object):

    def __init__(self, iou_threshold: float = 0.5, max_frames_count_no_detections: int = 15):
        """
        :param iou_threshold: bounding boxes IOU threshold when objects considered the same [0, 1.0]
        :param max_frames_count_no_detections: indicates number of frames to keep object alive if no detections exists for this object
        """
        self._iou_treshold = iou_threshold
        self._max_frames_count_no_detections = max_frames_count_no_detections

        self._counter = 0
        self._tracks = {}
        self._previous_frame_id = -1

    def process(self, frame_id: int, detections: List[ObjectInfo]) -> Tuple[List[ObjectInfo], List[int]]:
        """ Assigns track_id to each detection.
            Returns objects that are alive as tracks, and objects that disappeared in current frame

            :param frame_id: incremental frame offset [0, maxint)
        """

        # update tracks
        actives = []
        if self._previous_frame_id < frame_id:
            actives = self._update(frame_id, detections)
        else:
            raise ValueError("Failed. Previous frame id ({}) > Next({})".format(self._previous_frame_id,
                                                                                frame_id))

        # check existing tracks
        objects_to_remove = []
        for key, tracked_object in self._tracks.items():

            # check if we already updated track in detections
            # so there is no need to update this track twice
            if frame_id == tracked_object.last_frame_update:
                continue

            # if object exists
            if self._is_alive(tracked_object, frame_id):
                actives.append(tracked_object.object_info)
            else:
                objects_to_remove.append(tracked_object.idx)

        for idx in objects_to_remove:
            tracked_object = self._tracks.pop(idx, None)

        return actives, objects_to_remove

    def _is_alive(self, track: TrackedObject, frame_id: int) -> bool:
        return track.last_frame_update > 0 and \
            abs(track.last_frame_update - frame_id) < self._max_frames_count_no_detections

    def _update(self, frame_id: int, detections: List[ObjectInfo]) -> List[ObjectInfo]:
        """
            Updates tracks with detections for current frame (frame_id)
            Set "track_id" field in object info

            :param frame_id: incremental frame offset [0, maxint)
        """
        self._previous_frame_id = frame_id

        if not detections:
            return []

        ious = np.zeros(len(detections))
        used = set()  # prevent from using same detections in tracker update
        actives = []
        for track_id in self._tracks:
            for i, object_info in enumerate(detections):
                if i in used:
                    continue

                ious[i] = intersection_over_union(object_info.bounding_box,
                                                  self._tracks[track_id].bounding_box)

            # Find best match for detected-tracked object
            sorted_indices = np.argsort(ious)[::-1]
            best_index = sorted_indices[0]
            if ious[best_index] > self._iou_treshold:
                object_info = detections[best_index]
                self._tracks[track_id].update_from_detection(
                    frame_id, object_info)

                used.add(best_index)  # prevent from duplicating same object
                actives.append(object_info)
            ious.fill(0)  # reset calculated IOUS

        # create new tracked objects from detections
        # for those whose haven't been in tracked objects yet
        for i, object_info in enumerate(detections):
            if i in used:
                continue

            new_id = self._next_id
            self._tracks[new_id] = TrackedObject(
                idx=new_id, frame_id=frame_id, object_info=object_info)
            actives.append(object_info)
        return actives

    @property
    def _next_id(self) -> int:
        """ Returns unique track id """
        self._counter += 1
        return self._counter

    def __del__(self):
        self._tracks.clear()


class TrackerIOUAdapter(IModule):

    def __init__(self, iou_threshold: float = 0.5,
                 max_frames_count_no_detections: float = 15):
        """
            :param iou_threshold: identifies Intersection-Over-Union threshold when objects matches
            :max_frames_count_no_detections: max number of frames for track to be alive without any detections
        """
        super(TrackerIOUAdapter, self).__init__()
        self._tracker = IouObjectTracker(
            iou_threshold, max_frames_count_no_detections)

    def process(self, frame_data: FrameData, **kwargs) -> FrameData:

        frame_data.actives, frame_data.non_actives = self._tracker.process(
                frame_data.frame_offset, frame_data.actives)

        return frame_data
