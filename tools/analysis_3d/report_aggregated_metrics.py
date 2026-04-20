"""Generate a markdown report and bar-chart plots from aggregated_metrics.json.

Takes a list of (model_name, model_version, json_path) entries and produces,
for each location/vehicle_type, a subfolder containing:
  - A markdown report with per-label AP, APH, prediction counts, and GT counts.
  - Grouped bar-chart PNG figures (one per metric type).

Usage:
    # Save reports + plots to subfolders under work_dirs/metric_reports
    python3 tools/analysis_3d/report_aggregated_metrics.py --output-dir work_dirs/metric_reports

    # Only center_distance_bev
    python3 tools/analysis_3d/report_aggregated_metrics.py --output-dir work_dirs/metric_reports --metric-type center_distance_bev

    # Print to stdout (no plots)
    python3 tools/analysis_3d/report_aggregated_metrics.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Input specification ──────────────────────────────────────────────────────
# Each entry: (model_name, model_version, json_file_path)
json_files: list[tuple[str, str, str]] = [
    ("BEVFusion-LiDAR", "base/2.6.0", "work_dirs/bevfusion_lidar_2.6.0/base/T4Dataset/lidar_voxel_second_secfpn_50e_8xb8_base_120m_t4metric_v2/20260414_015859/testing/j6gen2/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "base/2.6.0", "work_dirs/bevfusion_lidar_2.6.0/base/T4Dataset/lidar_voxel_second_secfpn_50e_8xb8_base_120m_t4metric_v2/20260414_043334/testing/largebus/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "base/2.6.0", "work_dirs/bevfusion_lidar_2.6.0/base/T4Dataset/lidar_voxel_second_secfpn_50e_8xb8_base_120m_t4metric_v2/20260414_052656/testing/jpntaxi_gen2/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "base/2.6.0", "work_dirs/bevfusion_lidar_2.6.0/base/T4Dataset/lidar_voxel_second_secfpn_50e_8xb8_base_120m_t4metric_v2/20260414_110622/testing/base/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "j6gen2_base/2.6.1", "work_dirs/bevfusion_lidar_intensity_2.6.1/j6gen2_base/T4Dataset/lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m_t4metric_v2/20260415_145838/testing/largebus/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "j6gen2_base/2.6.1", "work_dirs/bevfusion_lidar_intensity_2.6.1/j6gen2_base/T4Dataset/lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m_t4metric_v2/20260415_154546/testing/j6gen2/aggregated_metrics.json"),
    ("BEVFusion-LiDAR", "j6gen2_base/2.6.1", "work_dirs/bevfusion_lidar_intensity_2.6.1/j6gen2_base/T4Dataset/lidar_voxel_second_secfpn_30e_8xb8_j6gen2_base_120m_t4metric_v2/20260415_181748/testing/j6gen2_base/aggregated_metrics.json"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate markdown report and plots from aggregated_metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for output. Sub-folders are created per "
            "location/vehicle_type, each containing markdown + plots. "
            "If omitted, the markdown report is printed to stdout (no plots)."
        ),
    )
    parser.add_argument(
        "--metric-type",
        choices=["center_distance_bev", "plane_distance", "all"],
        default="all",
        help="Which distance metric family to report (default: all).",
    )
    return parser.parse_args()


def _split_range_key(range_key: str) -> tuple[str, str]:
    """Split 'location/vehicle_type/metric_range' into (location/vehicle, metric_range)."""
    idx = range_key.rfind("/")
    if idx == -1:
        return ("", range_key)
    return (range_key[:idx], range_key[idx + 1 :])


def _extract_label_ap(
    label_metrics: dict,
    label: str,
    metric_type: str,
) -> float | None:
    """Compute the mean AP across distance thresholds for a label and metric type."""
    pattern = re.compile(
        rf"T4MetricV2_label/{re.escape(label)}_AP_{re.escape(metric_type)}_[\d.]+$"
    )
    values = [v for k, v in label_metrics.items() if pattern.match(k)]
    if not values:
        return None
    return sum(values) / len(values)


def _extract_label_aph(
    label_metrics: dict,
    label: str,
    metric_type: str,
) -> float | None:
    """Compute the mean APH across distance thresholds for a label and metric type."""
    pattern = re.compile(
        rf"T4MetricV2_label/{re.escape(label)}_APH_{re.escape(metric_type)}_[\d.]+$"
    )
    values = [v for k, v in label_metrics.items() if pattern.match(k)]
    if not values:
        return None
    return sum(values) / len(values)


def _extract_label_preds_gts(
    metadata_label: dict,
    label: str,
) -> tuple[int | None, int | None]:
    preds = metadata_label.get(f"metadata_label/test_{label}_num_predictions")
    gts = metadata_label.get(f"metadata_label/test_{label}_num_ground_truths")
    return preds, gts


def _fmt(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_int(val: int | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:,}"


THRESHOLDS = {
    "center_distance_bev": ["0.5", "1.0", "2.0", "4.0"],
    "plane_distance": ["2.0", "4.0"],
}


def _get_per_threshold(
    label_metrics: dict,
    label: str,
    metric_type: str,
    metric_name: str,
) -> list[float | None]:
    """Return values at each threshold for a given metric (AP, max-f1score, etc.)."""
    thresholds = THRESHOLDS.get(metric_type, [])
    values: list[float | None] = []
    for t in thresholds:
        key = f"T4MetricV2_label/{label}_{metric_name}_{metric_type}_{t}"
        values.append(label_metrics.get(key))
    return values


def _fmt_threshold_vals(vals: list[float | None], decimals: int = 3) -> str:
    parts = [f"{v:.{decimals}f}" if v is not None else "N/A" for v in vals]
    return " / ".join(parts)


def _safe_folder_name(loc_vehicle: str) -> str:
    return loc_vehicle.replace("/", "_")


# ── Data loading ─────────────────────────────────────────────────────────────


def _load_data(
    json_files: list[tuple[str, str, str]],
) -> tuple[
    dict[tuple[str, str], list[dict]],
    list[str],
    dict[str, list[str]],
]:
    """Load all JSON files and return grouped data.

    Returns:
        groups:  (metric_range, loc_vehicle) -> [entry dicts]
        metric_ranges_seen: ordered list of unique metric ranges
        loc_vehicles_seen:  metric_range -> ordered list of unique loc/vehicles
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    metric_ranges_seen: list[str] = []
    loc_vehicles_seen: dict[str, list[str]] = defaultdict(list)

    for model_name, model_version, fpath in json_files:
        path = Path(fpath)
        if not path.exists():
            print(f"Warning: file not found, skipping: {fpath}", file=sys.stderr)
            continue

        with open(path) as f:
            data = json.load(f)

        for range_key, block in data.items():
            loc_vehicle, metric_range = _split_range_key(range_key)

            if metric_range not in metric_ranges_seen:
                metric_ranges_seen.append(metric_range)
            if loc_vehicle not in loc_vehicles_seen[metric_range]:
                loc_vehicles_seen[metric_range].append(loc_vehicle)

            groups[(metric_range, loc_vehicle)].append(
                {
                    "model_name": model_name,
                    "model_version": model_version,
                    "metrics": block.get("metrics", {}),
                    "label_metrics": block.get("aggregated_metric_label", {}),
                    "metadata": block.get("metadata", {}),
                    "metadata_label": block.get("metadata_label", {}),
                }
            )

    return groups, metric_ranges_seen, loc_vehicles_seen


# ── Markdown report (per location/vehicle) ───────────────────────────────────


def build_location_report(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
) -> str:
    metric_type_display = metric_type.replace("_", " ").title()
    mAP_key = f"T4MetricV2/mAP_{metric_type}"
    mAPH_key = f"T4MetricV2/mAPH_{metric_type}"

    lines: list[str] = []
    lines.append(f"# {loc_vehicle} — {metric_type_display}")
    lines.append("")

    for metric_range in sorted(metric_ranges):
        key = (metric_range, loc_vehicle)
        if key not in groups:
            continue
        entries = groups[key]

        lines.append(f"## {metric_range}")
        lines.append("")

        all_labels: list[str] = []
        for entry in entries:
            for label in entry["label_metrics"]:
                if label not in all_labels:
                    all_labels.append(label)

        # Build header: label columns show name + GT count
        label_gts: dict[str, int] = {}
        for label in all_labels:
            for entry in entries:
                ml = entry["metadata_label"].get(label, {})
                g = ml.get(f"metadata_label/test_{label}_num_ground_truths")
                if g is not None:
                    label_gts[label] = g
                    break

        header_cols = ["Model version", "mAP", "mAPH"]
        for label in all_labels:
            gts = label_gts.get(label, 0)
            header_cols.append(f"{label}<br>({gts:,})")
        lines.append("| " + " | ".join(header_cols) + " |")

        sep_cols = [":----", "---:", "---:"] + ["---:"] * len(all_labels)
        lines.append("| " + " | ".join(sep_cols) + " |")

        # One row per model
        for entry in entries:
            m = entry["metrics"]
            model_id = f"{entry['model_name']} {entry['model_version']}"
            mAP_val = _fmt(m.get(mAP_key))
            mAPH_val = _fmt(m.get(mAPH_key))

            cells = [f"{model_id}", mAP_val, mAPH_val]
            for label in all_labels:
                lm = entry["label_metrics"].get(label, {})
                ml = entry["metadata_label"].get(label, {})
                ap = _extract_label_ap(lm, label, metric_type)
                preds, _ = _extract_label_preds_gts(ml, label)
                ap_str = _fmt(ap)
                preds_str = _fmt_int(preds)
                cells.append(f"{ap_str}<br>(preds: {preds_str})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # ── Detailed per-model tables ────────────────────────────────────
        thresholds = THRESHOLDS.get(metric_type, [])
        thresh_str = "/".join(thresholds)

        for entry in entries:
            model_id = f"{entry['model_name']} {entry['model_version']}"
            lines.append(f"**{model_id}**")
            lines.append("")
            lines.append(
                f"| Label | GTs | Preds | mAP "
                f"| AP@{thresh_str} "
                f"| max_f1@{thresh_str} "
                f"| optimal_conf@{thresh_str} |"
            )
            lines.append(
                "| :---- | ---: | ---: | ---: | :---- | :---- | :---- |"
            )

            m = entry["metrics"]
            mAP_val = _fmt(m.get(mAP_key))

            for label in all_labels:
                lm = entry["label_metrics"].get(label, {})
                ml = entry["metadata_label"].get(label, {})
                preds, gts = _extract_label_preds_gts(ml, label)

                ap_vals = _get_per_threshold(lm, label, metric_type, "AP")
                f1_vals = _get_per_threshold(lm, label, metric_type, "max-f1score")
                conf_vals = _get_per_threshold(lm, label, metric_type, "optimal-confidence")

                valid_aps = [v for v in ap_vals if v is not None]
                label_map = _fmt(sum(valid_aps) / len(valid_aps)) if valid_aps else "N/A"

                lines.append(
                    f"| {label} "
                    f"| {_fmt_int(gts)} "
                    f"| {_fmt_int(preds)} "
                    f"| {label_map} "
                    f"| {_fmt_threshold_vals(ap_vals)} "
                    f"| {_fmt_threshold_vals(f1_vals)} "
                    f"| {_fmt_threshold_vals(conf_vals)} |"
                )

            # Summary row
            total_gts = entry["metadata"].get(
                "metadata/test_total_num_ground_truths"
            )
            total_preds = entry["metadata"].get(
                "metadata/test_total_num_predictions"
            )
            lines.append(
                f"| **ALL** "
                f"| {_fmt_int(total_gts)} "
                f"| {_fmt_int(total_preds)} "
                f"| {mAP_val} "
                f"| — | — | — |"
            )
            lines.append("")

    return "\n".join(lines)


# ── Bar-chart plot (per location/vehicle) ────────────────────────────────────


def _get_label_gts(
    entries: list[dict],
    label: str,
) -> int:
    """Sum GTs for a label across entries (use first model's value since GTs are shared)."""
    for entry in entries:
        ml = entry["metadata_label"].get(label, {})
        gts = ml.get(f"metadata_label/test_{label}_num_ground_truths")
        if gts is not None:
            return gts
    return 0


def _get_total_gts(entries: list[dict]) -> int:
    for entry in entries:
        gts = entry["metadata"].get("metadata/test_total_num_ground_truths")
        if gts is not None:
            return gts
    return 0


def generate_location_plot(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
    out_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    mAP_key = f"T4MetricV2/mAP_{metric_type}"
    mAPH_key = f"T4MetricV2/mAPH_{metric_type}"
    metric_type_short = metric_type.replace("_", " ").title()

    sorted_ranges = sorted(metric_ranges)
    ranges_with_data = [
        r for r in sorted_ranges if (r, loc_vehicle) in groups
    ]
    if not ranges_with_data:
        return None

    n_subplots = len(ranges_with_data)
    fig, axes = plt.subplots(
        n_subplots, 1, figsize=(14, 5.5 * n_subplots), squeeze=False
    )

    for row, metric_range in enumerate(ranges_with_data):
        ax = axes[row, 0]
        entries = groups[(metric_range, loc_vehicle)]

        all_labels: list[str] = []
        for entry in entries:
            for label in entry["label_metrics"]:
                if label not in all_labels:
                    all_labels.append(label)

        label_gts = {lb: _get_label_gts(entries, lb) for lb in all_labels}
        total_gts = _get_total_gts(entries)
        x_tick_labels = [
            f"{lb}\n(GTs: {label_gts[lb]:,})" for lb in all_labels
        ] + [f"mAP\n(GTs: {total_gts:,})", "mAPH"]

        model_ids: list[str] = []
        for entry in entries:
            mid = f"{entry['model_name']}\n{entry['model_version']}"
            if mid not in model_ids:
                model_ids.append(mid)

        n_groups = len(x_tick_labels)
        n_models = len(model_ids)
        if n_models == 0 or n_groups == 0:
            continue

        x = np.arange(n_groups)
        bar_width = 0.8 / n_models

        for model_idx, mid in enumerate(model_ids):
            values = []
            for label in all_labels:
                ap_vals = []
                for entry in entries:
                    eid = f"{entry['model_name']}\n{entry['model_version']}"
                    if eid != mid:
                        continue
                    lm = entry["label_metrics"].get(label, {})
                    ap = _extract_label_ap(lm, label, metric_type)
                    if ap is not None:
                        ap_vals.append(ap)
                values.append(ap_vals[0] if ap_vals else 0.0)

            for agg_key in [mAP_key, mAPH_key]:
                agg_vals = []
                for entry in entries:
                    eid = f"{entry['model_name']}\n{entry['model_version']}"
                    if eid != mid:
                        continue
                    v = entry["metrics"].get(agg_key)
                    if v is not None:
                        agg_vals.append(v)
                values.append(agg_vals[0] if agg_vals else 0.0)

            offset = (model_idx - (n_models - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, values, bar_width, label=mid, zorder=3
            )
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("AP / mAP / mAPH", fontsize=12)
        ax.set_title(
            f"{metric_range}  ({metric_type_short})",
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(loc_vehicle, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ap_{metric_type}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    metric_types: list[str]
    if args.metric_type == "all":
        metric_types = ["center_distance_bev", "plane_distance"]
    else:
        metric_types = [args.metric_type]

    groups, metric_ranges_seen, loc_vehicles_seen = _load_data(json_files)

    all_loc_vehicles: list[str] = sorted(
        {lv for lvs in loc_vehicles_seen.values() for lv in lvs}
    )

    if args.output_dir:
        for loc_vehicle in all_loc_vehicles:
            sub_dir = args.output_dir / _safe_folder_name(loc_vehicle)
            sub_dir.mkdir(parents=True, exist_ok=True)

            for mt in metric_types:
                report = build_location_report(
                    loc_vehicle, groups, metric_ranges_seen, mt
                )
                md_path = sub_dir / f"report_{mt}.md"
                md_path.write_text(report)
                print(f"  {md_path}")

                png_path = generate_location_plot(
                    loc_vehicle, groups, metric_ranges_seen, mt, sub_dir
                )
                if png_path:
                    print(f"  {png_path}")
    else:
        for mt in metric_types:
            for loc_vehicle in all_loc_vehicles:
                report = build_location_report(
                    loc_vehicle, groups, metric_ranges_seen, mt
                )
                print(report)


if __name__ == "__main__":
    main()
