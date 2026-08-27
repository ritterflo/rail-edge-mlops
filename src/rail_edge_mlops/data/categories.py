"""nuImages' 25 fine-grained categories collapsed to the nuScenes 10 detection classes.

This is the official nuScenes detection taxonomy, used verbatim so that mAP here is
comparable with published numbers rather than being a figure only this repo can produce.

Categories absent from the mapping are dropped, not silently relabelled: animals, debris,
bicycle racks, strollers, wheelchairs, personal-mobility devices and emergency vehicles are
too rare to learn and would contribute noise to mAP. The conversion counts every dropped
annotation and reports the total, so "dropped" never means "unnoticed".
"""

from __future__ import annotations

NUIMAGES_TO_DETECTION: dict[str, str] = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

# Fixed order -> stable category ids across every rebuild of the dataset.
#
# ZERO-indexed, deliberately. COCO conventionally numbers categories from 1, but these
# ids are handed to the model as `class_labels`, which index a tensor of width
# num_labels. A 1-indexed id equal to num_labels reads one element past the end of the
# classification logits, inside a GPU kernel -- surfacing as an opaque
# HSA_STATUS_ERROR_EXCEPTION rather than an IndexError, and only when a batch happens
# to contain the last class.
DETECTION_CLASSES: list[str] = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(DETECTION_CLASSES)}
