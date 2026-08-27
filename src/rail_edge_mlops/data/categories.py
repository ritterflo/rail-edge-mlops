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

# Fixed order -> stable COCO category ids across every rebuild of the dataset.
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

CLASS_TO_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(DETECTION_CLASSES)}
