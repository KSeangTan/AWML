import copy
from typing import Any, Dict, Optional, Set

import mmengine
import webdataset as wds
from mmengine.logging import print_log
from mmengine.registry import DATASETS

from autoware_ml.detection3d.datasets.t4dataset import T4Dataset


@DATASETS.register_module()
class T4WebDataset(wds.WebDataset):
    """Streaming webdataset for T4 3-D detection.

    Inherits from ``wds.WebDataset`` for native tar-shard streaming, shard
    shuffling, and multi-worker splitting.  A ``T4Dataset`` instance is
    created internally to handle annotation parsing, filtering, and the
    detection pipeline.

    Each raw sample read from the tar is:

    1. Checked against the set of valid keys that survived
       ``T4Dataset.filter_data()`` — skipped if filtered out.
    2. Matched to its annotation via ``__key__`` (= positional index in the
       pickle ``data_list``).
    3. Enriched with binary sensor payloads from the tar.
    4. Passed through the ``T4Dataset`` pipeline transforms.

    Binary payloads are injected under the following keys so that
    downstream pipeline transforms can read from memory instead of disk:

    * ``data_info["lidar_points"]["lidar_bytes"]``
    * ``data_info["lidar_sweeps"][i]["lidar_points"]["lidar_bytes"]``
    * ``data_info["images"][<cam>]["img_bytes"]``

    Args:
        wds_path: Glob / brace-expansion pattern for tar shards,
            e.g. ``"/data/shards/train/*.tar"``
        shardshuffle: Number of shards to shuffle, or ``True`` for all.
        shuffle_buffer: In-memory shuffle buffer size.  0 disables.
        shuffle_seed: Base seed for deterministic shuffling.  Combined
            with an auto-incrementing epoch counter so both shard order
            and sample order differ every epoch.
        metainfo: Forwarded to ``T4Dataset``.
        class_names: Forwarded to ``T4Dataset``.
        use_valid_flag: Forwarded to ``T4Dataset``.
        **kwargs: Forwarded to ``T4Dataset`` (and ``NuScenesDataset``).
    """

    def __init__(
        self,
        wds_path: str,
        shardshuffle: bool = True,
        shuffle_buffer: int = 1000,
        shuffle_seed: int = 0,
        metainfo=None,
        class_names=None,
        use_valid_flag: bool = False,
        **kwargs,
    ):
        self.t4_dataset = T4Dataset(
            metainfo=metainfo,
            class_names=class_names,
            use_valid_flag=use_valid_flag,
            **kwargs,
        )
        if not self.t4_dataset._fully_initialized:
            self.t4_dataset.full_init()

        self._raw_data_list = mmengine.load(self.t4_dataset.ann_file)["data_list"]
        self._valid_keys = self._build_valid_keys()

        # detshuffle=True: shard order is reshuffled every epoch with a
        # deterministic seed so results are reproducible across runs.
        super().__init__(
            wds_path,
            shardshuffle=shardshuffle,
            detshuffle=True,
            seed=shuffle_seed,
        )
        self.select(lambda s: int(s["__key__"]) in self._valid_keys)
        if shuffle_buffer > 0 and not self.t4_dataset.test_mode:
            # Sample-level detshuffle: auto-increments its epoch counter
            # each time the iterator restarts, reseeding the RNG so the
            # sample order is different every epoch.
            self.append(wds.detshuffle(shuffle_buffer, seed=shuffle_seed))
        self.map(self._process_sample)
        self.select(lambda x: x is not None)

        num_filtered = len(self._raw_data_list) - len(self._valid_keys)
        print_log(
            f"T4WebDataset: {len(self._valid_keys)} valid samples "
            f"({num_filtered} filtered), streaming from {wds_path}",
            logger="current",
        )

    def _build_valid_keys(self) -> Set[int]:
        """Cross-reference filtered data_list with the raw pickle to find
        which original indices (= tar ``__key__`` values) survived filtering.
        """
        valid_tokens = set()
        for idx in range(len(self.t4_dataset)):
            info = self.t4_dataset.data_list[idx]
            valid_tokens.add(info["sample_key"])

        return valid_tokens

    def __len__(self) -> int:
        return len(self._valid_keys)

    def _process_sample(self, wds_sample: dict) -> Optional[Dict[str, Any]]:
        """Transform a raw tar sample into a pipeline-processed example.

        Returns ``None`` for samples that should be skipped (filtered key,
        empty ground truth, or failed pipeline).
        """
        key = int(wds_sample["__key__"])
        if key not in self._valid_keys:
            return None

        data_info = self.t4_dataset.get_data_info(key)
        self._inject_wds_data(data_info, wds_sample)

        data_info["box_type_3d"] = self.t4_dataset.box_type_3d
        data_info["box_mode_3d"] = self.t4_dataset.box_mode_3d

        if not self.t4_dataset.test_mode and self.t4_dataset.filter_empty_gt:
            ann_info = data_info.get("ann_info", {})
            gt_labels = ann_info.get("gt_labels_3d")
            if gt_labels is None or len(gt_labels) == 0:
                return None

        example = self.t4_dataset.pipeline(data_info)
        if example is None:
            return None

        if not self.t4_dataset.test_mode and self.t4_dataset.filter_empty_gt:
            if len(example["data_samples"].gt_instances_3d.labels_3d) == 0:
                return None

        return example

    def _inject_wds_data(self, data_info: dict, wds_sample: dict) -> None:
        """Inject pre-loaded binary payloads from a tar sample.

        Adds raw bytes under ``*_bytes`` keys so that pipeline transforms
        can read sensor data directly from memory instead of disk.
        """
        if "lidar.pcd" in wds_sample:
            data_info["lidar_points"]["lidar_bytes"] = wds_sample["lidar.pcd"]

        for i, sweep in enumerate(data_info.get("lidar_sweeps", [])):
            sweep_key = f"lidar_sweep_{i}.pcd"
            if sweep_key in wds_sample:
                sweep["lidar_points"]["lidar_bytes"] = wds_sample[sweep_key]

        for cam_name in list(data_info.get("images", {}).keys()):
            cam_lower = cam_name.lower()
            for ext in ("jpg", "jpeg", "png"):
                wds_key = f"{cam_lower}.{ext}"
                if wds_key in wds_sample:
                    data_info["images"][cam_name]["img_bytes"] = wds_sample[wds_key]
                    break
