custom_imports = dict(
    imports=[
        "autoware_ml.detection3d.datasets.t4dataset",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric",
        "autoware_ml.detection3d.evaluation.t4metric.t4metric_v2",
    ]
)

# dataset type setting
dataset_type = "T4Dataset"
info_train_file_name = "t4dataset_j6gen2_base_infos_train.pkl"
info_val_file_name = "t4dataset_j6gen2_base_infos_val.pkl"
info_test_file_name = "t4dataset_j6gen2_base_infos_test.pkl"

info_train_statistics_file_name = "t4dataset_j6gen2_base_statistics_train.parquet"
info_val_statistics_file_name = "t4dataset_j6gen2_base_statistics_val.parquet"
info_test_statistics_file_name = "t4dataset_j6gen2_base_statistics_test.parquet"

# dataset scene setting
dataset_version_list = [
    "db_j6gen2_v1",
    "db_j6gen2_v2",
    "db_j6gen2_v3",
    "db_j6gen2_v4",
    "db_j6gen2_v5",
    "db_j6gen2_v6",
    "db_j6gen2_v7",
    "db_j6gen2_v8",
    "db_j6gen2_v9",
    "db_largebus_v1",
    "db_largebus_v2",
    "db_largebus_v3",
]

dataset_test_groups = {
    "largebus": ("t4dataset_largebus_infos_test.pkl", False),
    "j6gen2": ("t4dataset_j6gen2_infos_test.pkl", False),
    "j6gen2_base": ("t4dataset_j6gen2_base_infos_test.pkl", True),
}

# dataset format setting
data_prefix = dict(
    pts="",
    CAM_FRONT="",
    CAM_FRONT_LEFT="",
    CAM_FRONT_RIGHT="",
    CAM_BACK="",
    CAM_BACK_RIGHT="",
    CAM_BACK_LEFT="",
    sweeps="",
)

camera_types = {
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
}

# class setting
name_mapping = {
    # CAR
    "ambulance": "car",
    "car": "car",
    "forklift": "car",
    "kart": "car",
    "other_vehicle": "car",
    "police_car": "car",
    # TRUCK
    "construction_vehicle": "truck",
    "tractor": "truck",
    "tractor_unit": "truck",
    "truck": "truck",
    "fire_truck": "truck",
    # TRAILER
    "semi_trailer": "trailer",
    "vehicle.trailer": "trailer",
    "trailer": "trailer",
    # BUS
    "bus": "bus",
    # BICYCLE
    "bicycle": "bicycle",
    "motorcycle": "bicycle",
    # PEDESTRIAN
    "pedestrian": "pedestrian",
    "construction_worker": "pedestrian",
    "other_pedestrian": "pedestrian",
    "personal_mobility": "pedestrian",
    "stroller": "pedestrian",
    "wheelchair": "pedestrian",
    "police_officer": "pedestrian",
    # BARRIER
    "barrier": "barrier",
    # TRAFFIC CONE
    "traffic_cone": "traffic_cone",
    # ANIMAL
    "animal": "animal",
    # Construction sign
    "construction_sign": "construction_sign",
    "construction sign": "construction_sign",
    # train
    "train": "train",
}

class_names = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "pedestrian",
]
num_class = len(class_names)
metainfo = dict(classes=class_names)

merge_objects = [
    ("truck", ["truck", "trailer"]),
]
merge_type = "extend_longer"  # One of ["extend_longer","union", None]

# visualization
class_colors = {
    "car": (30, 144, 255),
    "truck": (140, 0, 255),
    "construction_vehicle": (255, 255, 0),
    "bus": (111, 255, 111),
    "trailer": (0, 255, 255),
    "barrier": (0, 0, 0),
    "motorcycle": (100, 0, 30),
    "bicycle": (255, 0, 30),
    "pedestrian": (255, 200, 200),
    "traffic_cone": (120, 120, 120),
}
camera_panels = [
    "data/CAM_FRONT_LEFT",
    "data/CAM_FRONT",
    "data/CAM_FRONT_RIGHT",
    "data/CAM_BACK_LEFT",
    "data/CAM_BACK",
    "data/CAM_BACK_RIGHT",
]

filter_attributes = []

evaluator_metric_configs = dict(
    evaluation_task="detection",
    target_labels=class_names,
    center_distance_bev_thresholds=[0.5, 1.0, 2.0, 4.0],
    # plane_distance_thresholds is required for the pass fail evaluation
    plane_distance_thresholds=[2.0, 4.0],
    iou_2d_thresholds=None,
    iou_3d_thresholds=None,
    label_prefix="autoware",
    # bev minimum distance ranges for each range bucket, must be the same length as max_distance,
    # they will form bev distance ranges in [(min_distance[0], max_distance[0]), (min_distance[1], max_distance[1]), ...] when filtering
    min_distance=[0.0, 50.0, 90.0, 0.0],
    # bev maximum distance ranges for each range bucket, must be the same length as min_distance
    max_distance=[50.0, 90.0, 121.0, 121.0],
    min_point_numbers=0,
    matching_class_agnostic_fps=False,
)
