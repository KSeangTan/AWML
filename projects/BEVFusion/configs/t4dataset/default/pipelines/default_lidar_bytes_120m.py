_base_ = [
    "./default_lidar_120m.py",
]

train_pipeline = [
    dict(
        type="LoadPointsFromBytes",
        coord_type="LIDAR",
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_load_dim,
        lidar_point_bytes_key="lidar_point_bytes",
    ),
    dict(
        type="LoadPointsBytesFromMultiSweeps",
        sweeps_num=_base_.sweeps_num,
        load_dim=_base_.point_load_dim,
        use_dim=_base_.lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        lidar_point_bytes_key="lidar_point_bytes",
        test_mode=False,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="BEVFusionGlobalRotScaleTrans",
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.2],
    ),
    dict(type="BEVFusionRandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=_base_.point_cloud_range),
    dict(
        type="ObjectNameFilter",
        classes=[
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
    ),
    dict(type="PointShuffle"),
    dict(
        type="Pack3DDetInputs",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "timestamp",
        ],
    ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=point_load_dim,
        use_dim=point_load_dim,
        backend_args=backend_args,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=point_load_dim,
        use_dim=lidar_sweep_dims,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
        test_mode=True,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="Pack3DDetInputs",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "img_path",
            "num_pts_feats",
            "num_views",
            "timestamp",
        ],
    ),
]

# Filtering configuration
# Note:
# - In camera–LiDAR configs, `filter_cfg` can enable image-based frame filtering,
#   e.g., dict(filter_frames_with_missing_image=True).
# - This is a LiDAR-only config (`input_modality['use_camera'] = False`), so
#   image-based filtering does not apply and `filter_cfg` is intentionally None.
filter_cfg = None
