"""Plot histograms of LiDAR point counts for matched predictions and false
negatives (unmatched ground truths) from evaluator.pkl."""

from __future__ import annotations

import argparse
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

MATCHING_MODE_ALIASES = {
    "center_distance": MatchingMode.CENTERDISTANCE,
    "center_distance_bev": MatchingMode.CENTERDISTANCEBEV,
    "plane_distance": MatchingMode.PLANEDISTANCE,
}

# Percentiles (as percentages) to compute and annotate for each subplot.
STAT_PERCENTILES = [25, 50, 75, 80, 90, 95, 98, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load evaluator.pkl and plot histograms of pointcloud_num for matched "
            "predictions and false-negative (unmatched) ground truths at a given "
            "matching threshold."
        )
    )
    parser.add_argument(
        "evaluator_pkl",
        type=Path,
        help="Path to evaluator.pkl saved by T4MetricV2.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save figures. Defaults to the directory containing evaluator.pkl.",
    )
    parser.add_argument(
        "--matching-mode",
        type=str,
        default="center_distance_bev",
        choices=sorted(MATCHING_MODE_ALIASES.keys()),
        help="Matching mode used to select matched prediction/GT pairs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Matching threshold in meters.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional upper cap for histogram x-axis.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saved PNG files.",
    )
    return parser.parse_args()


def _resolve_threshold_key(threshold_map: Dict[float, List[DynamicObjectWithPerceptionResult]], threshold: float):
    if threshold in threshold_map:
        return threshold
    for key in threshold_map:
        if math.isclose(float(key), threshold, rel_tol=0.0, abs_tol=1e-6):
            return key
    return None


def _is_matched_pair(
    object_result: DynamicObjectWithPerceptionResult,
    matching_mode: MatchingMode,
    threshold: float,
) -> bool:
    if object_result.ground_truth_object is None:
        return False
    return object_result.is_result_correct(
        matching_mode=matching_mode,
        matching_threshold=threshold,
    )


def _is_unmatched_gt(object_result: DynamicObjectWithPerceptionResult) -> bool:
    """A false negative: a ground truth that has no associated prediction.

    NOTE: this assumes the object_result entries in the swept threshold_map
    include one-sided entries for ground truths that were never assigned an
    estimated_object (mirroring the one-sided FP entries that have
    ground_truth_object=None). If your evaluator.pkl does not produce such
    one-sided entries, this will silently return zero false negatives -- see
    the warning printed in main() if that happens.
    """
    return object_result.ground_truth_object is not None and object_result.estimated_object is None


def collect_matched_pointcloud_counts(
    evaluators: dict,
    matching_mode: MatchingMode,
    threshold: float,
) -> Tuple[DefaultDict[str, DefaultDict[str, List[int]]], DefaultDict[str, DefaultDict[str, List[int]]]]:
    """Collect pointcloud_num for matched predictions and their paired GT boxes."""
    pred_counts: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    gt_counts: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for evaluator_name, evaluator_data in evaluators.items():
        frame_results = evaluator_data.perception_evaluator_manager.frame_results
        for frame_result in frame_results:
            nuscene_object_results = frame_result.nuscene_object_results
            if nuscene_object_results is None:
                continue

            mode_results = nuscene_object_results.get(matching_mode)
            if mode_results is None:
                continue

            for label, threshold_map in mode_results.items():
                class_name = label.value
                threshold_key = _resolve_threshold_key(threshold_map, threshold)
                if threshold_key is None:
                    continue

                for object_result in threshold_map[threshold_key]:
                    if not _is_matched_pair(object_result, matching_mode, threshold):
                        continue

                    pred_num = int(object_result.estimated_object.pointcloud_num)
                    gt_num = int(object_result.ground_truth_object.pointcloud_num)
                    if pred_num == 0 or gt_num == 0:
                        continue

                    pred_counts[evaluator_name][class_name].append(pred_num)
                    gt_counts[evaluator_name][class_name].append(gt_num)

    return pred_counts, gt_counts


def collect_false_negative_counts(
    evaluators: dict,
    matching_mode: MatchingMode,
    threshold: float,
) -> DefaultDict[str, DefaultDict[str, List[int]]]:
    """Collect pointcloud_num for false-negative (unmatched) ground truths.

    Ground truths with pointcloud_num == 0 are excluded.
    """
    fn_counts: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for evaluator_name, evaluator_data in evaluators.items():
        frame_results = evaluator_data.perception_evaluator_manager.frame_results
        for frame_result in frame_results:
            nuscene_object_results = frame_result.nuscene_object_results
            if nuscene_object_results is None:
                continue

            mode_results = nuscene_object_results.get(matching_mode)
            if mode_results is None:
                continue

            for label, threshold_map in mode_results.items():
                class_name = label.value
                threshold_key = _resolve_threshold_key(threshold_map, threshold)
                if threshold_key is None:
                    continue

                for object_result in threshold_map[threshold_key]:
                    if not _is_unmatched_gt(object_result):
                        continue

                    gt_num = int(object_result.ground_truth_object.pointcloud_num)
                    if gt_num == 0:
                        continue

                    fn_counts[evaluator_name][class_name].append(gt_num)

    return fn_counts


def _sorted_classes(class_names: Iterable[str]) -> List[str]:
    return sorted(set(class_names))


def compute_stats(counts: np.ndarray) -> Dict[str, float]:
    """Compute min, percentiles, max, mean, and variance for an array of counts."""
    stats: Dict[str, float] = {
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
        "mean": float(np.mean(counts)),
        "var": float(np.var(counts)),
    }
    for p in STAT_PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(counts, p))
    return stats


def _format_stats_text(stats: Dict[str, float]) -> str:
    lines = [
        f"min={stats['min']:.0f}  max={stats['max']:.0f}",
        f"mean={stats['mean']:.1f} \u00b1 {stats['var']:.1f} (var)",
        f"p25={stats['p25']:.0f}  p50={stats['p50']:.0f}  p75={stats['p75']:.0f}",
        f"p80={stats['p80']:.0f}  p90={stats['p90']:.0f}  p95={stats['p95']:.0f}",
        f"p98={stats['p98']:.0f}  p99={stats['p99']:.0f}",
    ]
    return "\n".join(lines)


def plot_histograms(
    counts_by_evaluator: DefaultDict[str, DefaultDict[str, List[int]]],
    output_dir: Path,
    matching_mode: MatchingMode,
    threshold: float,
    bins: int,
    max_points: int | None,
    dpi: int,
    *,
    title_label: str,
    filename_prefix: str,
    xlabel: str,
    no_data_label: str,
) -> Tuple[List[Path], DefaultDict[str, DefaultDict[str, Dict[str, float]]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    all_stats: DefaultDict[str, DefaultDict[str, Dict[str, float]]] = defaultdict(dict)

    for evaluator_name, class_to_counts in sorted(counts_by_evaluator.items()):
        class_names = _sorted_classes(class_to_counts.keys())
        if not class_names:
            continue

        fig, axes = plt.subplots(
            nrows=len(class_names),
            ncols=1,
            figsize=(10, 3.8 * len(class_names)),
            squeeze=False,
        )
        fig.suptitle(
            f"{evaluator_name}\n"
            f"{title_label} ({matching_mode.value}, threshold={threshold:g} m)",
            fontsize=14,
            y=0.995,
        )

        for row_idx, class_name in enumerate(class_names):
            ax = axes[row_idx, 0]
            counts = np.asarray(class_to_counts[class_name], dtype=np.int32)
            if counts.size == 0:
                ax.set_title(f"{class_name} ({no_data_label})")
                ax.axis("off")
                continue

            stats = compute_stats(counts)
            all_stats[evaluator_name][class_name] = stats

            upper = max_points if max_points is not None else int(np.max(counts))
            hist_range = (0, max(upper, 1))
            ax.hist(counts, bins=bins, range=hist_range, color="#1f77b4", edgecolor="white", linewidth=0.5)
            ax.set_xlim(hist_range)
            ax.set_yscale("log")
            ax.set_ylim(bottom=0.8)  # keeps bins with count=1 visible instead of touching the axis floor
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=45))
            ax.set_title(f"{class_name} (n={counts.size})")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count (log scale)")
            ax.grid(axis="y", which="both", alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

            ax.text(
                0.98,
                0.95,
                _format_stats_text(stats),
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="right",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.85),
            )

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        safe_name = evaluator_name.replace("/", "_")
        output_path = output_dir / f"{filename_prefix}_{safe_name}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths, all_stats


def _print_stats(title: str, counts: DefaultDict[str, DefaultDict[str, List[int]]], all_stats: DefaultDict[str, DefaultDict[str, Dict[str, float]]]) -> None:
    print(f"\n{title} pair counts / statistics:")
    for evaluator_name in sorted(counts.keys()):
        print(f"  [{evaluator_name}]")
        class_names = _sorted_classes(counts[evaluator_name].keys())
        for class_name in class_names:
            n = len(counts[evaluator_name][class_name])
            if n == 0 or class_name not in all_stats.get(evaluator_name, {}):
                print(f"    {class_name}: n={n}")
                continue
            s = all_stats[evaluator_name][class_name]
            print(
                f"    {class_name}: n={n} min={s['min']:.0f} p25={s['p25']:.0f} p50={s['p50']:.0f} "
                f"p75={s['p75']:.0f} p80={s['p80']:.0f} p90={s['p90']:.0f} p95={s['p95']:.0f} "
                f"p98={s['p98']:.0f} p99={s['p99']:.0f} max={s['max']:.0f} "
                f"mean={s['mean']:.1f}+-{s['var']:.1f}(var)"
            )


def main() -> None:
    args = parse_args()
    matching_mode = MATCHING_MODE_ALIASES[args.matching_mode]
    output_dir = args.output_dir or args.evaluator_pkl.parent / "matched_pred_pointcloud_hist_normalized"

    with open(args.evaluator_pkl, "rb") as f:
        evaluators = pickle.load(f)

    if not isinstance(evaluators, dict):
        raise TypeError(f"Expected evaluator.pkl to contain a dict, got {type(evaluators)}")

    pred_counts, gt_counts = collect_matched_pointcloud_counts(
        evaluators=evaluators,
        matching_mode=matching_mode,
        threshold=args.threshold,
    )
    fn_counts = collect_false_negative_counts(
        evaluators=evaluators,
        matching_mode=matching_mode,
        threshold=args.threshold,
    )

    total_matches = sum(len(values) for class_map in pred_counts.values() for values in class_map.values())
    if total_matches == 0:
        available_modes = set()
        for evaluator_data in evaluators.values():
            for frame_result in evaluator_data.perception_evaluator_manager.frame_results:
                if frame_result.nuscene_object_results is not None:
                    available_modes.update(frame_result.nuscene_object_results.keys())
        raise RuntimeError(
            "No matched prediction/GT pairs found. "
            f"Requested matching_mode={matching_mode.value}, threshold={args.threshold}. "
            f"Available matching modes in pickle: {[mode.value for mode in sorted(available_modes, key=lambda m: m.value)]}"
        )

    total_fn = sum(len(values) for class_map in fn_counts.values() for values in class_map.values())
    if total_fn == 0:
        print(
            "WARNING: No false-negative (unmatched ground truth) entries were found. "
            "This likely means evaluator.pkl does not represent unmatched ground truths as "
            "object_result entries with estimated_object=None -- see the docstring of "
            "_is_unmatched_gt() and adjust the FN-detection logic to match your data model."
        )

    saved_paths, pred_stats = plot_histograms(
        counts_by_evaluator=pred_counts,
        output_dir=output_dir,
        matching_mode=matching_mode,
        threshold=args.threshold,
        bins=args.bins,
        max_points=args.max_points,
        dpi=args.dpi,
        title_label="Matched predictions",
        filename_prefix="matched_pred_pointcloud_hist",
        xlabel="num_pointclouds in matched prediction bbox",
        no_data_label="no matched predictions",
    )

    fn_saved_paths, fn_stats = plot_histograms(
        counts_by_evaluator=fn_counts,
        output_dir=output_dir,
        matching_mode=matching_mode,
        threshold=args.threshold,
        bins=args.bins,
        max_points=args.max_points,
        dpi=args.dpi,
        title_label="False negatives (unmatched ground truths)",
        filename_prefix="false_negative_pointcloud_hist",
        xlabel="num_pointclouds in unmatched GT bbox",
        no_data_label="no false negatives",
    )

    all_saved_paths = saved_paths + fn_saved_paths
    print(f"Saved {len(all_saved_paths)} figure(s) to: {output_dir}")
    for path in all_saved_paths:
        print(f"  - {path}")

    print("\nMatched pair counts (prediction pointcloud_num samples / GT pointcloud_num samples):")
    for evaluator_name in sorted(pred_counts.keys()):
        print(f"  [{evaluator_name}]")
        class_names = _sorted_classes(pred_counts[evaluator_name].keys())
        for class_name in class_names:
            pred_n = len(pred_counts[evaluator_name][class_name])
            gt_n = len(gt_counts[evaluator_name][class_name])
            print(f"    {class_name}: pred={pred_n}, gt={gt_n}")

    _print_stats("Prediction pointcloud_num", pred_counts, pred_stats)
    _print_stats("False-negative GT pointcloud_num", fn_counts, fn_stats)


if __name__ == "__main__":
    main()