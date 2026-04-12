_base_ = [
    "../../../../../autoware_ml/configs/detection3d/default_runtime.py",
    "../../../../../autoware_ml/configs/detection3d/dataset/t4dataset/largebus.py",
    "../default/pipelines/default_lidar_intensity_bytes_120m.py",
    "../default/models/default_lidar_second_secfpn_120m.py",
    "../default/schedulers/default_30e_4xb8_adamw_cosine.py",
    "../default/default_misc.py",
]

custom_imports = dict(imports=["projects.BEVFusion.bevfusion"], allow_failed_imports=False)
custom_imports["imports"] += _base_.custom_imports["imports"]
custom_imports["imports"] += ["autoware_ml.detection3d.datasets.transforms"]

# user setting
data_root = "data/t4datasets/"
info_directory_path = "info/kokseang_test/"

experiment_group_name = "bevfusion_lidar_intensity/largebus/" + _base_.dataset_type
experiment_name = "lidar_voxel_second_secfpn_30e_4xb8_largebus_120m_webdataset"
work_dir = "work_dirs/" + experiment_group_name + "/" + experiment_name
webdataset_data_root = data_root + "webdataset/largebus_v1/"

# model parameter
model = dict(
    type="BEVFusion",
    voxelize_cfg=dict(
        point_cloud_range=_base_.point_cloud_range,
        voxel_size=_base_.voxel_size,
        voxelize_reduce=True,
    ),
    pts_voxel_encoder=dict(num_features=_base_.point_use_dim),
    pts_middle_encoder=dict(
        in_channels=_base_.point_use_dim,
        sparse_shape=_base_.grid_size,
    ),
    bbox_head=dict(
        class_names=_base_.class_names,  # Use class names to identify the correct class indices
        train_cfg=dict(
            point_cloud_range=_base_.point_cloud_range,
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size,
        ),
        test_cfg=dict(
            grid_size=_base_.grid_size,
            voxel_size=_base_.voxel_size[0:2],
            pc_range=_base_.point_cloud_range[0:2],
        ),
        bbox_coder=dict(
            pc_range=_base_.point_cloud_range[0:2],
            voxel_size=_base_.voxel_size[0:2],
        ),
    ),
)

# Dataset parameters
train_dataloader = dict(
    batch_size=_base_.train_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    # sampler=dict(type="DefaultSampler", shuffle=True),
    sampler=None,
    dataset=dict(
        type="T4WebDataset",
        data_root=data_root,
        wds_path=webdataset_data_root + "train/",
        glob_wds_path=True,
        shards_shuffle_buffer=30,
        samples_shuffle_buffer=400,
        shuffle_seed=_base_.randomness_seed,
        load_imgs=False,
        load_lidars=True,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        use_valid_flag=False,
        ann_file=info_directory_path + _base_.info_train_file_name,
        test_mode=False,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        filter_cfg=_base_.filter_cfg,
        pipeline=_base_.train_pipeline,
        backend_args=_base_.backend_args,
        empty_check=True,
	),
)

val_dataloader = dict(
    batch_size=_base_.test_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    # sampler=dict(type="DefaultSampler", shuffle=True),
    sampler=None,
    dataset=dict(
        type="T4WebDataset",
        data_root=data_root,
        wds_path=webdataset_data_root + "val/",
        shards_shuffle_buffer=30,
        glob_wds_path=True,
        samples_shuffle_buffer=200,
        shuffle_seed=_base_.randomness_seed,
        load_imgs=False,
        load_lidars=True,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        use_valid_flag=False,
        ann_file=info_directory_path + _base_.info_val_file_name,
        test_mode=True,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        filter_cfg=_base_.filter_cfg,
        pipeline=_base_.test_pipeline,
        backend_args=_base_.backend_args,
        empty_check=False,
	),
)

test_dataloader = dict(
    batch_size=_base_.test_batch_size,
    num_workers=_base_.num_workers,
    persistent_workers=True,
    # sampler=dict(type="DefaultSampler", shuffle=True),
    sampler=None,
    dataset=dict(
        type="T4WebDataset",
        data_root=data_root,
        wds_path=webdataset_data_root + "test/",
        glob_wds_path=True,
        shards_shuffle_buffer=30,
        samples_shuffle_buffer=200,
        shuffle_seed=_base_.randomness_seed,
        load_imgs=False,
        load_lidars=True,
        metainfo=_base_.metainfo,
        class_names=_base_.class_names,
        use_valid_flag=False,
        ann_file=info_directory_path + _base_.info_train_file_name,
        test_mode=True,
        data_prefix=_base_.data_prefix,
        box_type_3d="LiDAR",
        filter_cfg=_base_.filter_cfg,
        pipeline=_base_.test_pipeline,
        backend_args=_base_.backend_args,
        empty_check=False,
	),
)

# val_dataloader = dict(
#     batch_size=_base_.test_batch_size,
#     num_workers=_base_.num_workers,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset=dict(
#         type=_base_.dataset_type,
#         data_root=data_root,
#         ann_file=info_directory_path + _base_.info_val_file_name,
#         pipeline=_base_.test_pipeline,
#         metainfo=_base_.metainfo,
#         class_names=_base_.class_names,
#         modality=_base_.input_modality,
#         data_prefix=_base_.data_prefix,
#         test_mode=True,
#         box_type_3d="LiDAR",
#         backend_args=_base_.backend_args,
#     ),
# )

# test_dataloader = dict(
#     batch_size=_base_.test_batch_size,
#     num_workers=_base_.num_workers,
#     persistent_workers=True,
#     sampler=dict(type="DefaultSampler", shuffle=False),
#     dataset=dict(
#         type=_base_.dataset_type,
#         data_root=data_root,
#         ann_file=info_directory_path + _base_.info_test_file_name,
#         pipeline=_base_.test_pipeline,
#         metainfo=_base_.metainfo,
#         class_names=_base_.class_names,
#         modality=_base_.input_modality,
#         data_prefix=_base_.data_prefix,
#         test_mode=True,
#         box_type_3d="LiDAR",
#         backend_args=_base_.backend_args,
#     ),
# )

val_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_val_file_name,
    metric="bbox",
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=_base_.eval_class_range,
    filter_attributes=_base_.filter_attributes,
)

test_evaluator = dict(
    type="T4Metric",
    data_root=data_root,
    ann_file=data_root + info_directory_path + _base_.info_test_file_name,
    metric="bbox",
    backend_args=_base_.backend_args,
    class_names=_base_.class_names,
    name_mapping=_base_.name_mapping,
    eval_class_range=_base_.eval_class_range,
    filter_attributes=_base_.filter_attributes,
    save_csv=True,
)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="NuScenes metric/T4Metric/mAP"),
)
log_processor = dict(window_size=50)
