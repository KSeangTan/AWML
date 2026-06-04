"""
The script is taken from mmdetection3d and
modified to create gt databases for 3D bounding boxes.
Note that we only support T4Dataset for now.
"""

from pathlib import Path

import numpy as np
from mmdet3d.structures.ops import box_np_ops
from mmengine import print_log, track_iter_progress
from mmengine.registry import DATASETS


class GTDatabasesCreator:
    """Given the raw data, generate the ground truth database. This is the
    parallel version. For serialized version, please refer to
    `create_groundtruth_database`

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
        used_classes (list[str], optional): Classes have been used.
            Default: None.
        database_save_path (str, optional): Path to save database.
            Default: None.
        db_info_save_path (str, optional): Path to save db_info.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(
        self,
        dataset_class_name,
        data_path,
        info_prefix,
        info_path=None,
        used_classes=None,
        database_save_path=None,
        db_info_save_path=None,
        relative_path=True,
        num_worker=8,
    ) -> None:
        self.dataset_class_name = dataset_class_name
        self.data_path = data_path
        self.info_prefix = info_prefix
        self.info_path = info_path
        self.used_classes = used_classes
        self.database_save_path = Path(database_save_path)
        self.database_save_path.mkdir(parents=True, exist_ok=True)

        self.db_info_save_path = Path(db_info_save_path)
        self.num_worker = num_worker
        self.pipeline = None

    def create_single(self, input_dict):
        group_counter = 0
        single_db_infos = dict()
        example = self.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].numpy()
        names = [self.dataset.metainfo["classes"][i] for i in annos["gt_labels_3d"]]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        for i in range(num_obj):
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = self.database_save_path / filename
            rel_filepath = f"{self.info_prefix}_gt_database" / filename

            # save point clouds for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (self.used_classes is None) or names[i] in self.used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in single_db_infos:
                    single_db_infos[names[i]].append(db_info)
                else:
                    single_db_infos[names[i]] = [db_info]

        return single_db_infos

    def create(self):
        print_log(f"Create GT Database of {self.dataset_class_name}", logger="current")
        dataset_cfg = dict(type=self.dataset_class_name, data_root=self.data_path, ann_file=self.info_path)
        if self.dataset_class_name == "T4Dataset":
            dataset_cfg.update(
                use_valid_flag=True,
                data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="sweeps/LIDAR_TOP"),
                pipeline=[
                    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                ],
            )
        else:
            raise ValueError(f"Unsupported dataset class name: {self.dataset_class_name}")

        self.dataset = DATASETS.build(dataset_cfg)
        self.pipeline = self.dataset.pipeline
        if self.database_save_path is None:
            self.database_save_path = osp.join(self.data_path, f"{self.info_prefix}_gt_database")
        if self.db_info_save_path is None:
            self.db_info_save_path = osp.join(self.data_path, f"{self.info_prefix}_dbinfos_train.pkl")
        mmengine.mkdir_or_exist(self.database_save_path)
        if self.with_mask:
            self.coco = COCO(osp.join(self.data_path, self.mask_anno_path))
            imgIds = self.coco.getImgIds()
            self.file2id = dict()
            for i in imgIds:
                info = self.coco.loadImgs([i])[0]
                self.file2id.update({info["file_name"]: i})

        def loop_dataset(i):
            input_dict = self.dataset.get_data_info(i)
            input_dict["box_type_3d"] = self.dataset.box_type_3d
            input_dict["box_mode_3d"] = self.dataset.box_mode_3d
            return input_dict

        if self.num_worker == 0:
            multi_db_infos = mmengine.track_progress(
                self.create_single, ((loop_dataset(i) for i in range(len(self.dataset))), len(self.dataset))
            )
        else:
            multi_db_infos = mmengine.track_parallel_progress(
                self.create_single,
                ((loop_dataset(i) for i in range(len(self.dataset))), len(self.dataset)),
                self.num_worker,
                chunksize=1000,
            )
        print_log("Make global unique group id", logger="current")
        group_counter_offset = 0
        all_db_infos = dict()
        for single_db_infos in track_iter_progress(multi_db_infos):
            group_id = -1
            for name, name_db_infos in single_db_infos.items():
                for db_info in name_db_infos:
                    group_id = max(group_id, db_info["group_id"])
                    db_info["group_id"] += group_counter_offset
                if name not in all_db_infos:
                    all_db_infos[name] = []
                all_db_infos[name].extend(name_db_infos)
            group_counter_offset += group_id + 1

        for k, v in all_db_infos.items():
            print_log(f"load {len(v)} {k} database infos", logger="current")

        print_log(f"Saving GT database infos into {self.db_info_save_path}")
        with open(self.db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)
