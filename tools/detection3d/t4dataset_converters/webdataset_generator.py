import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import webdataset as wds
from mmengine.logging import print_log


class WebDatasetGenerator:
    """
    Generator for webdataset samples from T4dataset.
    This will group all sensor data from a single frame into a single tar file and it will give the corresponding
    frame index in the pickle info file as the unique identifier.
    """

    def __init__(
        self, data_root_path: str, camera_types: Sequence[str], out_dir: str, max_scenes_per_shard: int, version: str
    ):
        self.data_root_path = Path(data_root_path)
        self.camera_types = camera_types
        self.out_dir = Path(out_dir)
        self.max_scenes_per_shard = max_scenes_per_shard
        self.version = version

    def read_lidar_bytes(self, scenario_path: Path, lidar_rel_path: str) -> bytes:
        """Read lidar point-cloud file and return raw bytes."""
        full_path = scenario_path / lidar_rel_path
        with open(full_path, "rb") as f:
            return f.read()

    def read_camera_bytes(self, scenario_path: Path, cam_rel_path: str) -> bytes:
        """Read camera image file and return raw bytes."""
        full_path = scenario_path / cam_rel_path
        with open(full_path, "rb") as f:
            return f.read()

    def get_camera_extension(self, cam_path: Path) -> str:
        """Return the file extension (without dot) for the camera image."""
        ext = cam_path.suffix.lstrip(".")
        if ext == "":
            return "jpg"
        return ext

    def build_wds_sample(
        self, sample_index: int, scenario_path: Path, camera_types: set, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a single webdataset sample dict.
        The dict maps extension-based keys to binary payloads:
          - ``__key__``          : unique frame identifier in the pickle info file.
          - ``lidar.pcd``        : raw lidar point-cloud bytes
          - ``<cam_name>.jpg``   : raw camera image bytes (per camera)
          - ``info.json``        : JSON-serialised annotation / metadata
        """
        wds_sample: Dict[str, Any] = {"__key__": sample_index}

        lidar_rel_path = info.get("lidar_points", {}).get("lidar_path")
        if not lidar_rel_path:
            raise ValueError(f"Lidar path not found in info: {info}")

        wds_sample["lidar.pcd"] = self.read_lidar_bytes(scenario_path, lidar_rel_path)
        # Add lidar sweeps
        lidar_sweeps = info.get("lidar_sweeps", [])
        for sweep_idx, sweep in enumerate(lidar_sweeps):
            wds_sample[f"lidar_sweep_{sweep_idx}.pcd"] = self.read_lidar_bytes(scenario_path, sweep["lidar_path"])

        for cam in self.camera_types:
            cam_info = info.get("images", {}).get(cam, {})
            cam_rel_path = cam_info.get("img_path")
            if not cam_rel_path:
                continue
            ext = self.get_camera_extension(cam_rel_path)
            cam_key = f"{cam.lower()}.{ext}"
            try:
                wds_sample[cam_key] = self.read_camera_bytes(scenario_path, cam_rel_path)
            except FileNotFoundError:
                print_log(f"Camera file not found: {cam_rel_path}", level=logging.WARNING)
                wds_sample[cam_key] = None

        return wds_sample

    def write_shards(
        self,
        scene_groups: Sequence[Sequence[Dict[str, Any]]],
        split: str,
    ) -> None:
        """Write scene-grouped webdataset samples into sharded tar files.

        Every scene's samples are written contiguously.  A new shard is opened
        every ``max_scenes_per_shard`` scenes, so samples from the same scene
        are never split across shards.

        Output pattern:  ``<out_dir>/<split>/t4dataset_<version>_<split>-%06d.tar``
        """
        split_dir = self.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_pattern = split_dir / f"t4dataset_{self.version}_{split}-%06d.tar"

        total_samples = 0
        shard_idx = 0
        scene_count_in_shard = 0
        sink = wds.TarWriter(shard_pattern % shard_idx)

        for scene_samples in scene_groups:
            if scene_count_in_shard > self.max_scenes_per_shard:
                sink.close()
                shard_idx += 1
                sink = wds.TarWriter(shard_pattern % shard_idx)
                scene_count_in_shard = 0

            for sample in scene_samples:
                sink.write(sample)
                total_samples += 1
            scene_count_in_shard += 1

        sink.close()
        num_shards = shard_idx + 1 if scene_groups else 0
        print_log(
            f"[{split}] Wrote {total_samples} samples from {len(scene_groups)} scenes "
            f"into {num_shards} shard(s) at {split_dir}",
            logger="current",
        )
