_base_ = [
    "./second_secfpn_8xb16_121m_base_amp.py",
]

# user setting
experiment_name = "resnet34_secfpn_8xb16_121m_base_amp_rfs_t4metric_v2"
work_dir = "work_dirs/" + _base_.experiment_group_name + "/" + experiment_name

train_frame_object_sampler = dict(
    type="FrameObjectSampler",
    object_samplers=[
        dict(
            type="ObjectBEVDistanceSampler",
            bev_distance_thresholds=[
                _base_.point_cloud_range[0],
                _base_.point_cloud_range[1],
                _base_.point_cloud_range[3],
                _base_.point_cloud_range[4],
            ],
        ),
        dict(
            type="LowPedestriansObjectSampler",
            height_threshold=1.5,
            bev_distance_thresholds=[
                -50.0,
                -50.0,
                50.0,
                50.0,
            ],
        ),
    ],
)

train_dataloader = dict(
    sampler=dict(type="DistributedWeightedRandomSampler", shuffle=True),
    dataset=dict(
        type="T4FrameSamplerDataset",
        repeat_sampling_factor=0.30,
        frame_object_sampler=train_frame_object_sampler,
    ),
)

model = dict(
	pts_backbone=dict(
		_delete_=True,
		type="BEVResNet",  # Use custom BEV-friendly ResNet wrapper (renamed to avoid confusion)
		depth=34,
		num_stages=3,
		strides=(1, 2, 2),  # ResNet stage strides: stage0=1, stage1=2, stage2=2
		dilations=(1, 1, 1),  # Dilation for each stage
		out_indices=(0, 1, 2),  # Get features from res_layers 0, 1, 2
		# BEV-friendly stem configuration: no downsampling at input
		deep_stem=True,  # Use three 3x3 convs instead of 7x7: more efficient and better boundary behavior
		conv1_stride=1,  # First conv stride=1 (no downsampling) - applies to deep_stem's first 3x3 conv
		with_pool=False,  # Disable maxpool (no downsampling)
		# pool_stride is only used when with_pool=True, so omitted here
		frozen_stages=-1,  # Don't freeze any stages initially
		base_channels=64,  # ResNet34 outputs: 64, 128, 256 channels (64*1, 64*2, 64*4)
		# ResNet34 uses BasicBlock (expansion=1), so base_channels=64 gives [64, 128, 256]
		norm_cfg=dict(
				type="BN", eps=1e-5, momentum=0.01
		),  # Fixed: eps changed from 1e-3 to 1e-5 for numerical stability
		norm_eval=False,  # Keep BN in training mode for better performance
		# Remove pretrained weights due to input channel mismatch (3 vs 32)
		# init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet34"),
		style="pytorch",
		in_channels=32,
		# pretrained=True,
		init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/resnet_34/resnet34_8xb32_mmcls.pth',
        prefix='backbone.'  # Often needed to map keys correctly
    ),
		with_cp=True,
	),
	pts_neck=dict(
		type="SECONDFPN",
		in_channels=[
				64,
				128,
				256,
		],  # ResNet34 layers 0, 1, 2: 64, 128, 256 channels (base_channels=64 * expansion=1 for BasicBlock)
		# Same as SECOND backbone: [64, 128, 256]
		out_channels=[128, 128, 128],
		# BEV-friendly: With conv1_stride=1 and no maxpool, outputs should be:
		# stage0: (1020, 1020) -> downsample stride=0.5 -> (510, 510)
		# stage1: (510, 510) -> upsample stride=1 -> (510, 510)
		# stage2: (255, 255) -> upsample stride=2 -> (510, 510)
		# Final output: (510, 510) to match target size (grid_size // out_size_factor)
		upsample_strides=[0.5, 1, 2],  # Upsample to match target feature map size (510, 510)
		norm_cfg=dict(
				type="BN", eps=1e-5, momentum=0.01
		),  # Fixed: eps changed from 0.001 (1e-3) to 1e-5 for numerical stability
		upsample_cfg=dict(type="deconv", bias=False),
		use_conv_for_no_stride=True,
  ),
	# pts_bbox_head=dict(
	# 		# sigmoid(-6.9078) = 0.01 for initial small values
	# 		separate_head=dict(type="CustomSeparateHead", init_bias=-6.9078, final_kernel=1),
  #   ),
)

max_epochs = 50
# learning rate
# Since mmengine doesn't support OneCycleMomentum yet, we use CosineAnnealing from the default configs
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first (max_epochs * 0.3) epochs, learning rate increases from 0 to lr * 10
    # during the next epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        T_max=int(max_epochs * 0.3),
        eta_min=lr * 10,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=lr * 1e-4,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    # momentum scheduler
    # During the first (0.3 * max_epochs) epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type="CosineAnnealingMomentum",
        T_max=int(max_epochs * 0.3),
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(max_epochs * 0.3),
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingMomentum",
        T_max=max_epochs - int(max_epochs * 0.3),
        eta_min=1,
        begin=int(max_epochs * 0.3),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

optimizer = dict(type="AdamW", lr=lr, weight_decay=0.01)
clip_grad = dict(max_norm=1.0, norm_type=2)  # max norm of gradients upper bound to be 15 since amp is used

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=optimizer,
    clip_grad=clip_grad,
    # Update it accordingly
    loss_scale={
        "init_scale": 2.0**8,  # intial_scale: 256
        "growth_interval": 2000,
    },
)

# Add evaluator configs
perception_evaluator_configs = dict(
    dataset_paths=_base_.data_root,
    frame_id="base_link",
    evaluation_config_dict=_base_.evaluator_metric_configs,
    load_raw_data=False,
)

frame_pass_fail_config = dict(
    target_labels=_base_.class_names,
    # Matching thresholds per class (must align with `plane_distance_thresholds` used in evaluation)
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0, 2.0],
    confidence_threshold_list=None,
)
training_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_train_statistics_file_name
)

testing_statistics_parquet_path = _base_.data_root + _base_.info_directory_path + _base_.info_test_statistics_file_name

validation_statistics_parquet_path = (
    _base_.data_root + _base_.info_directory_path + _base_.info_val_statistics_file_name
)

val_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_val_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="validation",
    dataset_name="base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=False,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
)

test_evaluator = dict(
    _delete_=True,
    type="T4MetricV2",
    data_root=_base_.data_root,
    ann_file=_base_.data_root + _base_.info_directory_path + _base_.info_test_file_name,
    training_statistics_parquet_path=training_statistics_parquet_path,
    testing_statistics_parquet_path=testing_statistics_parquet_path,
    validation_statistics_parquet_path=validation_statistics_parquet_path,
    output_dir="testing",
    dataset_name="base",
    perception_evaluator_configs=perception_evaluator_configs,
    critical_object_filter_config=None,
    frame_pass_fail_config=frame_pass_fail_config,
    num_workers=64,
    scene_batch_size=-1,
    write_metric_summary=True,
    class_names={{_base_.class_names}},
    name_mapping={{_base_.name_mapping}},
    experiment_name=experiment_name,
    experiment_group_name=_base_.experiment_group_name,
)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="T4MetricV2/T4MetricV2/mAP_center_distance_bev"
    ),
)


