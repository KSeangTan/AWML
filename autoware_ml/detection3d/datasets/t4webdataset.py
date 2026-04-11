import copy
import math
import random
from typing import Any, Dict, Optional, Set

import mmengine
from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from mmengine.registry import DATASETS
import webdataset as wds
from webdataset import filters as wds_filters, utils as wds_utils, shardlists


from autoware_ml.detection3d.datasets.t4dataset import T4Dataset


class DetSimpleShardList(shardlists.SimpleShardList):
    """A custom SimpleShardList that supports new epoch seed shuffling.
    
    This class is used to create a SimpleShardList that supports new epoch seed shuffling.
    """
    def __init__(self, urls, seed=None, epoch: int = -1):
        super().__init__(urls, seed)
        assert seed is not None, "Seed must be provided!"
        # Reset the seed value to make it consistent
        self.seed = seed 
        self.epoch = epoch

    def __len__(self):
        """Return the number of URLs in the list.

        Returns:
            int: The number of URLs.
        """
        return super().__len__()

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        self.epoch += 1
        urls = self.urls.copy()
        if self.seed is not None:
            seed = self.seed + self.epoch
            random.Random(seed).shuffle(urls)
        
        for url in urls:
            yield dict(url=url)


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
        shuffle_seed: Seed for deterministic shard shuffling via
            ``detshuffle``.
        repeat: Whether to call ``repeat()`` on the stream. Defaults to
            ``True`` in training and ``False`` in test mode.
        pad_ddp: If ``True``, compute per-rank epoch size with ceil
            division in DDP so all ranks run the same number of steps.
        split_by_node: If ``True``, enable WebDataset node-level shard
            partitioning (for multi-GPU / multi-node training).
        split_by_worker: If ``True``, enable WebDataset worker-level shard
            partitioning (for multi-worker dataloading).
        metainfo: Forwarded to ``T4Dataset``.
        class_names: Forwarded to ``T4Dataset``.
        use_valid_flag: Forwarded to ``T4Dataset``.
        **kwargs: Forwarded to ``T4Dataset`` (and ``NuScenesDataset``).
    """

    def __init__(
        self,
        wds_path: str,
        metainfo: dict,
        class_names: list,
        shuffle_seed: int,
        shards_shuffle_buffer: int | bool = 100,
        samples_shuffle_buffer: int = 1000,
        pad_ddp: bool = True,
        split_by_node: bool = True,
        split_by_worker: bool = True,
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
        self._rank, self._world_size = get_dist_info()
        self._global_num_samples = len(self._valid_keys)
        self._is_train = not self.t4_dataset.test_mode
        self._shuffle_seed = shuffle_seed

        self._samples_per_rank = self._global_num_samples
        if self._is_train:
            shards_shuffle_buffer = shards_shuffle_buffer if shards_shuffle_buffer > 0 else True
            if pad_ddp and self._world_size > 1:
                self._samples_per_rank = math.ceil(self._samples_per_rank / self._world_size)
                self._repeat = True
            else:
                self._repeat = False
                
            filter_stages = [
                wds_filters.select(lambda s: int(s["__key__"]) in self._valid_keys),
                wds_filters.map(self._process_sample),
                wds_filters.select(lambda x: x is not None),
                wds_filters.detshuffle(buffer=samples_shuffle_buffer, seed=self._shuffle_seed),
            ]
        else:
            shards_shuffle_buffer = None
            self._repeat = False
            filter_stages = [
                wds_filters.map(self._process_sample),
            ]

        super().__init__(
            wds_path,
            shardshuffle=shards_shuffle_buffer,
            detshuffle=True,
            seed=self._shuffle_seed,
            nodesplitter=wds.split_by_node if split_by_node else None,
            workersplitter=wds.split_by_worker if split_by_worker else None,
        )
        
        for stage in filter_stages:
            self.append(stage)

        # Repeat the stream if needed
        if self._repeat:
            self.repeat()
        
        # Set the number of samples per rank in training only
        if self._is_train:
            self.with_epoch(self._samples_per_rank)

        num_filtered = len(self._raw_data_list) - len(self._valid_keys)
        print_log(
            f"T4WebDataset: {len(self._valid_keys)} valid samples "
            f"({num_filtered} filtered), streaming from {wds_path}; "
            f"rank/world={self._rank}/{self._world_size}, "
            f"repeat={self._repeat}, samples_per_rank={self._samples_per_rank}"
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
        return self._samples_per_rank

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

    def create_url_iterator(self, args):
        """Create an appropriate URL iterator based on the input type.

        This method determines the type of URL input and creates the corresponding
        iterator for the dataset.

        Args:
            args: A SimpleNamespace object containing the arguments.

        Raises:
            ValueError: If the URL type is not supported or implemented.
        """
        if isinstance(args.urls, str) or wds_utils.is_iterable(args.urls):
            if args.mode == "resampled":
                self.append(shardlists.ResampledShardList(args.urls))
            else:
                self.append(DetSimpleShardList(args.urls, seed=self._shuffle_seed))
            return

        return super().create_url_iterator(args)
        