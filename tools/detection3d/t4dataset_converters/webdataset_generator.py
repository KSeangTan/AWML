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

    Supports a streaming mode via :meth:`add_scene` / :meth:`close` so that
    binary sensor data is flushed to disk per-shard instead of being held
    entirely in memory.
    """

    def __init__(
        self, data_root_path: str, camera_types: Sequence[str], out_dir: Path, max_scenes_per_shard: int, version: str
    ):
        self.data_root_path = Path(data_root_path)
        self.camera_types = camera_types
        self.out_dir = out_dir
        self.max_scenes_per_shard = max_scenes_per_shard
        self.version = version

        self._writers: Dict[str, wds.TarWriter] = {}
        self._shard_indices: Dict[str, int] = {}
        self._scene_counts: Dict[str, int] = {}
        self._total_samples: Dict[str, int] = {}
        self._total_scenes: Dict[str, int] = {}

    def read_lidar_bytes(self, lidar_rel_path: str) -> bytes:
        """Read lidar point-cloud file and return raw bytes."""
        full_path = self.data_root_path / lidar_rel_path
        with open(full_path, "rb") as f:
            return f.read()

    def read_camera_bytes(self, cam_rel_path: str) -> bytes:
        """Read camera image file and return raw bytes."""
        full_path = self.data_root_path / cam_rel_path
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
        wds_sample: Dict[str, Any] = {"__key__": str(sample_index)}

        lidar_rel_path = info.get("lidar_points", {}).get("lidar_path")
        if not lidar_rel_path:
            raise ValueError(f"Lidar path not found in info: {info}")

        wds_sample["lidar.pcd"] = self.read_lidar_bytes(lidar_rel_path)
        # Add lidar sweeps
        lidar_sweeps = info.get("lidar_sweeps", [])
        for sweep_idx, sweep in enumerate(lidar_sweeps):
            wds_sample[f"lidar_sweep_{sweep_idx}.pcd"] = self.read_lidar_bytes(sweep["lidar_points"]["lidar_path"])

        for cam in self.camera_types:
            cam_info = info.get("images", {}).get(cam, {})
            cam_rel_path = cam_info.get("img_path")
            if not cam_rel_path:
                continue
            
            ext = self.get_camera_extension(Path(cam_rel_path))
            cam_key = f"{cam.lower()}.{ext}"
            try:
                wds_sample[cam_key] = self.read_camera_bytes(cam_rel_path)
            except FileNotFoundError:
                print_log(f"Camera file not found: {cam_rel_path}", level=logging.WARNING)

        return wds_sample

    def _shard_pattern(self, split: str) -> str:
        split_dir = self.out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        return str(split_dir / f"t4dataset_{self.version}_{split}-%06d.tar")

    def _get_writer(self, split: str) -> wds.TarWriter:
        """Return the active TarWriter for *split*, creating one if needed."""
        if split not in self._writers:
            self._shard_indices[split] = 0
            self._scene_counts[split] = 0
            self._total_samples[split] = 0
            self._total_scenes[split] = 0
            self._writers[split] = wds.TarWriter(self._shard_pattern(split) % 0)
        return self._writers[split]

    def _rotate_shard(self, split: str) -> None:
        """Close the current shard and open a new one."""
        self._writers[split].close()
        self._shard_indices[split] += 1
        self._writers[split] = wds.TarWriter(
            self._shard_pattern(split) % self._shard_indices[split]
        )
        self._scene_counts[split] = 0

    def add_scene(
        self,
        scene_samples: Sequence[Dict[str, Any]],
        split: str,
    ) -> None:
        """Write one scene's samples to the current shard, rotating if full.

        Samples from the same scene are always kept in the same shard.
        Binary payloads are flushed to disk immediately so memory can be
        reclaimed by the caller after this call returns.
        """
        if not scene_samples:
            return

        sink = self._get_writer(split)

        if self._scene_counts[split] >= self.max_scenes_per_shard:
            self._rotate_shard(split)
            sink = self._writers[split]

        for sample in scene_samples:
            sink.write(sample)
            self._total_samples[split] += 1
        self._scene_counts[split] += 1
        self._total_scenes[split] += 1

    def close(self, split: str) -> None:
        """Finalize the writer for *split* and log a summary."""
        if split not in self._writers:
            return
        self._writers[split].close()
        num_shards = self._shard_indices[split] + 1
        split_dir = self.out_dir / split
        print_log(
            f"[{split}] Wrote {self._total_samples[split]} samples from "
            f"{self._total_scenes[split]} scenes into {num_shards} shard(s) at {split_dir}",
            logger="current",
        )
        del self._writers[split]

    def close_all(self) -> None:
        """Finalize all open writers."""
        for split in list(self._writers.keys()):
            self.close(split)
