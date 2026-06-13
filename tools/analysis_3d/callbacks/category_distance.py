from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from mmengine.logging import print_log
from mmdet3d.structures.ops import box_np_ops

from tools.analysis_3d.callbacks.callback_interface import AnalysisCallbackInterface
from tools.analysis_3d.data_classes import AnalysisData, DatasetSplitName
from tools.analysis_3d.split_options import SplitOptions


class CategoryDistancePointCountAnalysisCallback(AnalysisCallbackInterface):
    """Compute object counts by points-in-bbox bins for each category and distance range."""

    def __init__(
        self,
        out_path: Path,
        distance_ranges: Optional[List[Tuple[float, float]]] = None,
        point_bins: Optional[List[int]] = None,
        analysis_dir: str = "category_distance_points",
        remapping_classes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the callback to analyze lidar points in bboxes by distance ranges.
        
        :param out_path: Path to save outputs.
        :param distance_ranges: List of distance range tuples (min, max) in meters.
            Default: [(0, 50), (50, 90), (90, float('inf'))].
        :param point_bins: Exact point-count bins to report, e.g. [0, 1, 2, 3, 4, 5].
        :param analysis_dir: Folder name to save outputs.
        :param remapping_classes: Set if compute frequency of every category after remapping.
        """
        super(CategoryDistancePointCountAnalysisCallback, self).__init__()
        self.out_path = out_path
        self.distance_ranges = distance_ranges or [(0, 50), (50, 90), (90, 124), (124, float('inf'))]
        self.point_bins = point_bins or [0, 1, 2, 3, 4, 5]
        self.analysis_dir = analysis_dir
        self.remapping_classes = remapping_classes

        self.full_output_path = self.out_path / self.analysis_dir
        self.full_output_path.mkdir(exist_ok=True, parents=True)

        self.analysis_file_name = "category_distance_object_count_by_points_{}.png"
        self.y_axis_label = "Category"
        self.x_axis_label = "Number of Objects"
        self.legend_loc = "upper right"

    def _get_distance_range_label(self, min_dist: float, max_dist: float) -> str:
        """Generate a label for the distance range."""
        max_label = "∞" if max_dist == float('inf') else f"{max_dist}"
        return f"[{min_dist}, {max_label}]"

    def _calculate_object_distance(self, box: object) -> float:
        """
        Calculate distance from origin to the box center.
        
        :param box: Detection box with center position.
        :return: Distance in meters.
        """
        # Get the center of the bounding box
        # Calculate L2 distance from origin (0, 0, 0)
        distance = np.sqrt(box.position[0] ** 2 + box.position[1] ** 2 + box.position[2] ** 2)
        return distance

    def _points_in_boxes(self, points: npt.NDArray[np.float32], boxes: List[object]) -> npt.NDArray[np.int64]:
        """
        Count points falling inside each bounding box.
        
        :param points: Point cloud array of shape (N, 3) or (N, 4+) with first 3 columns as xyz.
        :param boxes: List of Box3D objects.
        :return: Array of point counts for each box.
        """
        if points.size == 0 or len(boxes) == 0:
            return np.zeros(len(boxes), dtype=np.int64)

        bboxes_3d = np.asarray([(
          box.box.position[0], 
          box.box.position[1], 
          box.box.position[2], 
          box.box.shape.size[1], 
          box.box.shape.size[0],
          box.box.shape.size[2],
          box.box.rotation.yaw_pitch_roll[0]
        ) for box in boxes])
        indices = box_np_ops.points_in_rbbox(
            points[:, :3],
            bboxes_3d,
        )
        num_points_in_gt = indices.sum(0)
        return num_points_in_gt

    def _get_category_distance_object_stats(
        self,
        analysis_data: AnalysisData,
    ) -> Dict[str, Dict[str, Dict[int, int]]]:
        """
        Count objects by exact point-count bins in each bbox, grouped by distance and category.
        
        :param analysis_data: AnalysisData containing all scenario data.
        :return: Dict of {distance_range: {category_name: {point_bin: object_count}}}.
        """
        distance_range_stats: Dict[str, Dict[str, Dict[int, int]]] = {}
        for min_dist, max_dist in self.distance_ranges:
            range_label = self._get_distance_range_label(min_dist, max_dist)
            distance_range_stats[range_label] = defaultdict(lambda: defaultdict(int))

        # Iterate through all samples and boxes
        for scenario_data in analysis_data.scenario_data.values():
            for sample_data in scenario_data.sample_data.values():
                # Load lidar points if available
                lidar_path = sample_data.lidar_point.lidar_path
                data_root_path = Path(analysis_data.data_root_path)
                lidar_path = data_root_path / lidar_path    
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(
                    -1, sample_data.lidar_point.num_pts_feats)
                
                bboxes = sample_data.detection_boxes
                if not len(bboxes) or points is None:
                    continue

                num_points_in_bboxes = self._points_in_boxes(points, bboxes) 
                # Process each detection box
                for detection_box, num_points in zip(sample_data.detection_boxes, num_points_in_bboxes):
                    box_category_name = detection_box.box.semantic_label.name
                    
                    # Apply remapping if provided
                    if self.remapping_classes is not None:
                        box_category_name = self.remapping_classes.get(
                            box_category_name, box_category_name
                        )

                    # Calculate distance from origin to box center
                    distance = self._calculate_object_distance(detection_box.box)

                    # Determine which distance range this box falls into
                    for min_dist, max_dist in self.distance_ranges:
                        if min_dist <= distance < max_dist:
                            range_label = self._get_distance_range_label(min_dist, max_dist)
                            point_bin = int(num_points)
                            if point_bin in self.point_bins:
                                distance_range_stats[range_label][box_category_name][point_bin] += 1
                            break

        return distance_range_stats

    def _visualize_category_distance_object_stats(
        self,
        dataset_category_distance_stats: Dict[str, Dict[str, Dict[str, Dict[int, int]]]],
        split_name: str,
        figsize: Tuple[int, int] = (18, 12),
    ) -> None:
        """
        Visualize object counts by exact point-count bins for each category.
        
        :param dataset_category_distance_stats: Dict of {dataset_name: {distance_range: {category: {point_bin: object_count}}}}.
        :param split_name: Split name (train, test, val, etc.).
        :param figsize: Figure size.
        """
        # Collect all unique categories and distance ranges
        all_categories = set()
        all_distance_ranges = set()
        
        for distance_range_stats in dataset_category_distance_stats.values():
            all_distance_ranges.update(distance_range_stats.keys())
            for category_stats in distance_range_stats.values():
                all_categories.update(category_stats.keys())

        all_categories = sorted(list(all_categories))
        all_distance_ranges = sorted(list(all_distance_ranges))
        
        if not all_categories or not all_distance_ranges:
            print_log(f"No data available for {split_name}")
            return

        dataset_names = sorted(list(dataset_category_distance_stats.keys()))
        num_datasets = len(dataset_names)
        num_ranges = len(all_distance_ranges)
        _, axes = plt.subplots(nrows=num_datasets, ncols=num_ranges, figsize=figsize, squeeze=False)

        x = np.arange(len(all_categories))
        total_groups = len(self.point_bins)
        bar_width = 0.8 / max(total_groups, 1)

        for dataset_idx, dataset_name in enumerate(dataset_names):
            for range_idx, distance_range in enumerate(all_distance_ranges):
                ax = axes[dataset_idx][range_idx]
                range_stats = dataset_category_distance_stats[dataset_name].get(distance_range, {})

                for point_idx, point_bin in enumerate(self.point_bins):
                    counts = [range_stats.get(category, {}).get(point_bin, 0) for category in all_categories]
                    offset = (point_idx - (total_groups - 1) / 2) * bar_width
                    ax.bar(x + offset, counts, bar_width, label=f"points={point_bin}")

                if range_idx == 0:
                    ax.set_ylabel(f"{dataset_name}\n{self.x_axis_label}")
                if dataset_idx == num_datasets - 1:
                    ax.set_xlabel(self.y_axis_label)

                ax.set_title(f"{dataset_name} | {distance_range}")
                ax.set_xticks(x)
                ax.set_xticklabels(all_categories, rotation=35, ha="right")

                if dataset_idx == 0 and range_idx == num_ranges - 1:
                    ax.legend(loc=self.legend_loc, title="Points in bbox")

        plt.tight_layout()
        analysis_file_name = self.full_output_path / self.analysis_file_name.format(split_name)
        plt.savefig(
            fname=analysis_file_name,
            format="png",
            bbox_inches="tight",
        )
        plt.close()

    def run(self, dataset_split_analysis_data: Dict[DatasetSplitName, AnalysisData]) -> None:
        """Inherited, check the superclass."""
        print_log(f"Running {self.__class__.__name__}")
        for split_option in SplitOptions:
            dataset_category_distance_stats = {}
            
            for dataset_split_name, analysis_data in dataset_split_analysis_data.items():
                split_name = dataset_split_name.split_name
                if split_name != split_option.value:
                    continue

                dataset_name = dataset_split_name.dataset_version
                dataset_category_distance_stats[dataset_name] = self._get_category_distance_object_stats(
                    analysis_data=analysis_data
                )

            if dataset_category_distance_stats:
                self._visualize_category_distance_object_stats(
                    dataset_category_distance_stats=dataset_category_distance_stats,
                    split_name=split_option.value,
                )
        print_log(f"Done running {self.__class__.__name__}")
