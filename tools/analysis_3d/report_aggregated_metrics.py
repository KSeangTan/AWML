"""Generate a markdown report and bar-chart plots from aggregated_metrics.json.

Takes a list of (model_name, model_version, json_path) entries and produces,
for each location/vehicle_type, a subfolder containing:
  - A markdown report with per-label AP, APH, GT counts, TP error tables, and mean TP error tables.
  - Grouped bar-chart PNG figures (one per metric type).
  - Stacked bar charts for per-class TP errors (default, medium, optimal; one PNG each).
  - Grouped bar chart for mean-tp-error (default / medium / optimal columns; rows by range).

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
    # (
    #     "BEVFusion-LiDAR",
    #     "j6gen2_base/2.7.1",
    #     "/home/kokseang/results/bevfusion_lidar_2_7_1/j6gen2_base/evaluation/j6gen2/testing/j6gen2/aggregated_metrics.json"
    # ),
    # (
    #     "BEVFusion-LiDAR",
    #     "j6gen2_base/2.7.1",
    #     "/home/kokseang/results/bevfusion_lidar_2_7_1/j6gen2_base/evaluation/j6gen2_base/testing/j6gen2_base/aggregated_metrics.json"
    # ),
    # (
    #     "BEVFusion-LiDAR",
    #     "j6gen2_base/2.7.1",
    #     "/home/kokseang/results/bevfusion_lidar_2_7_1/j6gen2_base/evaluation/largebus/testing/largebus/aggregated_metrics.json"
    # ),
    # (
    #     "BEVFusion-LiDAR",
    #     "jpntaxi_base/2.7.1",
    #     "/home/kokseang/results/bevfusion_lidar_2_7_1/jpntaxi_base/evaluation/jpntaxi_gen2/testing/jpntaxi_gen2/aggregated_metrics.json"
    # ),
    (
        "BEVFusion-LiDAR",
        "base/2.8.0",
        "/home/kokseang/results/bevfusion_lidar_2_8_0/base/evaluation/base/testing/base/aggregated_metrics.json"    
    ),
    (
        "BEVFusion-LiDAR",
        "base/2.8.0",
        "/home/kokseang/results/bevfusion_lidar_2_8_0/base/evaluation/j6gen2/testing/j6gen2/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "base/2.8.0",
        "/home/kokseang/results/bevfusion_lidar_2_8_0/base/evaluation/j6gen2_base/testing/j6gen2_base/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "base/2.8.0",
        "/home/kokseang/results/bevfusion_lidar_2_8_0/base/evaluation/jpntaxi_gen2/testing/jpntaxi_gen2/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "base/2.8.0",
        "/home/kokseang/results/bevfusion_lidar_2_8_0/base/evaluation/largebus/testing/largebus/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "j6gen2_base/2.8.1",
        "/home/kokseang/results/bevfusion_lidar_2_8_1/j6gen2_base/evaluation/j6gen2/testing/j6gen2/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "j6gen2_base/2.8.1",
        "/home/kokseang/results/bevfusion_lidar_2_8_1/j6gen2_base/evaluation/j6gen2_base/testing/j6gen2_base/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "j6gen2_base/2.8.1",
        "/home/kokseang/results/bevfusion_lidar_2_8_1/j6gen2_base/evaluation/j6gen2_base/testing/j6gen2_base/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "j6gen2_base/2.8.1",
        "/home/kokseang/results/bevfusion_lidar_2_8_1/j6gen2_base/evaluation/largebus/testing/largebus/aggregated_metrics.json"
    ),
    (
        "BEVFusion-LiDAR",
        "jpntaxi_base/2.8.1",
        "/home/kokseang/results/bevfusion_lidar_2_8_1/jpntaxi_base/evaluation/jpntaxi_gen2/testing/jpntaxi_gen2/aggregated_metrics.json"
    )
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
    metric_label: str,
    metric_type: str,
) -> float | None:
    """Compute the mean AP across distance thresholds for a label and metric type."""
    pattern = re.compile(rf"T4MetricV2_label/{re.escape(metric_label)}_AP_{re.escape(metric_type)}_[\d.]+$")
    values = [v for k, v in label_metrics.items() if pattern.match(k)]
    if not values:
        return None
    return sum(values) / len(values)


def _extract_label_aph(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
) -> float | None:
    """Compute the mean APH across distance thresholds for a label and metric type."""
    pattern = re.compile(rf"T4MetricV2_label/{re.escape(metric_label)}_APH_{re.escape(metric_type)}_[\d.]+$")
    values = [v for k, v in label_metrics.items() if pattern.match(k)]
    if not values:
        return None
    return sum(values) / len(values)


def _tp_error_json_name(tp_error_type: str) -> str:
    return TP_ERROR_JSON_SUFFIX.get(tp_error_type, tp_error_type)


def _tp_error_key_fragment(tp_error_variant: str, tp_error_type: str) -> str:
    """Metric key fragment between label and matching_method (e.g. tp-error-medium-ATE)."""
    suffix = _tp_error_json_name(tp_error_type)
    if tp_error_variant == "medium":
        return f"tp-error-medium-{suffix}"
    if tp_error_variant == "optimal":
        return f"tp-error-optimal-{suffix}"
    return f"tp-error_{suffix}"


def _num_match_key_fragment(tp_error_variant: str | None) -> str:
    """JSON key fragment for num-match; None = summary table (plain num-match)."""
    if tp_error_variant is None:
        return "num-match"
    if tp_error_variant == "medium":
        return "medium-recall-num-match"
    if tp_error_variant == "optimal":
        return "optimal-num-match"
    return "min-recall-num-match"


def _num_match_column_label(_tp_error_variant: str, thresh_str: str) -> str:
    """Markdown column header (always num_match; JSON lookup uses hyphenated keys)."""
    return f"{NUM_MATCH_COLUMN_NAME}@{thresh_str}"


def _get_per_threshold_num_match(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
    tp_error_variant: str | None = None,
) -> list[float | None]:
    thresholds = THRESHOLDS.get(metric_type, [])
    fragment = _num_match_key_fragment(tp_error_variant)
    return [label_metrics.get(f"T4MetricV2_label/{metric_label}_{fragment}_{metric_type}_{t}") for t in thresholds]


def _extract_label_tp_error_mean(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
    tp_error_type: str,
    tp_error_variant: str = "",
) -> float | None:
    """Mean TP error across matching thresholds for one label and error type."""
    fragment = _tp_error_key_fragment(tp_error_variant, tp_error_type)
    pattern = re.compile(
        rf"T4MetricV2_label/{re.escape(metric_label)}_{re.escape(fragment)}_" rf"{re.escape(metric_type)}_[\d.]+$"
    )
    values = [v for k, v in label_metrics.items() if pattern.match(k)]
    if not values:
        return None
    return sum(values) / len(values)


def _extract_label_gts(
    metadata_label: dict,
    metric_label: str,
) -> int | None:
    return metadata_label.get(f"metadata_label/test_{metric_label}_num_ground_truths")


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

# Overall metrics kept in categorical summaries/charts.
# Each display label maps to candidate (bucket_name, metric_name) pairs.
# bucket_name=None reads from top-level metrics; otherwise from aggregated_metric_label.
CATEGORICAL_OVERALL_METRICS: tuple[tuple[str, tuple[tuple[str | None, str], ...]], ...] = (
    ("mAP", ((None, "mAP"),)),
    ("mAPH", ((None, "mAPH"),)),
    ("map_based_nds", ((None, "map_based_nds"), ("map", "map_based_nds"))),
    (
        "maph_based_nds",
        (
            (None, "mapH_based_nds"),
            (None, "maph_based_nds"),
            ("mapH", "mapH_based_nds"),
            ("maph", "maph_based_nds"),
            ("maph", "maph_based-nds"),
        ),
    ),
)

# Per-label TP error types (AEE in spec; JSON keys use AAE).
TP_ERROR_TYPES = ("ATE", "AOE", "ASE", "AVE", "AEE")
TP_ERROR_JSON_SUFFIX = {"AEE": "AAE"}
# Key variants: tp-error_{TYPE}, tp-error-medium-{TYPE}, tp-error-optimal-{TYPE}
TP_ERROR_CHART_VARIANTS: tuple[tuple[str, str], ...] = (
    ("", "default"),
    ("medium", "medium"),
    ("optimal", "optimal"),
)
# Markdown column header only (JSON keys remain hyphenated, e.g. num-match).
NUM_MATCH_COLUMN_NAME = "num_match"


def _tp_error_variant_display_title(tp_variant: str, variant_label: str, *, multiline: bool = False) -> str:
    """Display title with recall annotation for default / medium TP error variants."""
    if tp_variant == "":
        base, recall = "default", "recall @0.10"
    elif tp_variant == "medium":
        base, recall = "medium", "recall @0.40"
    else:
        return variant_label
    if multiline:
        return f"{base}\n{recall}"
    return f"{base} ({recall})"


def _mean_tp_column_title(tp_variant: str, variant_label: str) -> str:
    """Column header for mean TP bar chart (recall range annotation on default / medium)."""
    return _tp_error_variant_display_title(tp_variant, variant_label, multiline=True)


# Mean TP error spider axis order (clockwise from top: mATE → mAAE).
MEAN_TP_ERROR_TYPES = ("mATE", "mAOE", "mASE", "mAVE", "mAAE")


def _overall_metric_key(metric_name: str, matching_method: str) -> str:
    return f"T4MetricV2/{metric_name}_{matching_method}"


def _is_categorical_overall_label(label: str) -> bool:
    return label in {"map", "mapH", "maph"}


def _overall_metric_value(entry: dict, metric_label: str, matching_method: str) -> float | None:
    metrics = entry.get("metrics", {})
    label_metrics = entry.get("label_metrics", {})

    for display_label, metric_sources in CATEGORICAL_OVERALL_METRICS:
        if display_label != metric_label:
            continue

        for bucket_name, metric_name in metric_sources:
            source = metrics if bucket_name is None else label_metrics.get(bucket_name, {})
            value = source.get(_overall_metric_key(metric_name, matching_method))
            if value is not None:
                return value
        return None

    value = metrics.get(_overall_metric_key(metric_label, matching_method))
    if value is not None:
        return value

    legacy_label = metric_label.replace("maph_", "mapH_", 1)
    if legacy_label != metric_label:
        return metrics.get(_overall_metric_key(legacy_label, matching_method))
    return None


def _is_mean_tp_error_label(label: str) -> bool:
    """True for mean TP error aggregates in aggregated_metric_label (not object classes)."""
    return label.startswith("mean-tp-error")


def _get_per_threshold(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
    metric_name: str,
) -> list[float | None]:
    """Return values at each threshold for a given metric (AP, max-f1score, etc.)."""
    thresholds = THRESHOLDS.get(metric_type, [])
    values: list[float | None] = []
    for t in thresholds:
        key = f"T4MetricV2_label/{metric_label}_{metric_name}_{metric_type}_{t}"
        values.append(label_metrics.get(key))
    return values


def _get_per_threshold_tp_error(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
    tp_error_type: str,
    tp_error_variant: str = "",
) -> list[float | None]:
    """Return TP error values at each distance threshold for a label and error type."""
    thresholds = THRESHOLDS.get(metric_type, [])
    fragment = _tp_error_key_fragment(tp_error_variant, tp_error_type)
    values: list[float | None] = []
    for t in thresholds:
        key = f"T4MetricV2_label/{metric_label}_{fragment}_{metric_type}_{t}"
        values.append(label_metrics.get(key))
    return values


def _infer_metric_label(display_label: str, label_metrics: dict) -> str:
    """Infer the metric label used in metric keys for a given display label.

    Some aggregated_metrics.json files group metrics under a friendly label key
    (e.g. "traffic") while the actual metric keys use a different label
    (e.g. "traffic_cone"). We infer the real label by inspecting the inner keys.
    """
    # Example key:
    #   T4MetricV2_label/traffic_cone_AP_center_distance_bev_0.5
    pat = re.compile(
        r"^T4MetricV2_label/(?P<label>.+?)_(AP|APH|max-f1score|optimal-confidence|optimal-recall|optimal-precision)_.+$"
    )
    for k in label_metrics.keys():
        m = pat.match(k)
        if m:
            return m.group("label")
    return display_label


def _build_metric_label_index(entry: dict) -> dict[str, tuple[str, dict]]:
    """Map metric_label -> (bucket_label, label_metrics_dict) for one entry.

    aggregated_metrics.json sometimes stores a bucket label (e.g. "traffic")
    whose inner metric keys use a different label (e.g. "traffic_cone").
    This index lets us use the true metric label as the display/canonical label.
    """
    out: dict[str, tuple[str, dict]] = {}
    for bucket_label, lm in entry.get("label_metrics", {}).items():
        if _is_mean_tp_error_label(bucket_label):
            continue
        if _is_categorical_overall_label(bucket_label):
            continue
        metric_label = _infer_metric_label(bucket_label, lm)
        if _is_mean_tp_error_label(metric_label):
            continue
        if _is_categorical_overall_label(metric_label):
            continue
        out.setdefault(metric_label, (bucket_label, lm))
    return out


def _fmt_threshold_vals(vals: list[float | None], decimals: int = 3) -> str:
    parts = [f"{v:.{decimals}f}" if v is not None else "N/A" for v in vals]
    return " / ".join(parts)


def _fmt_threshold_int_vals(vals: list[float | None]) -> str:
    parts = []
    for v in vals:
        if v is None:
            parts.append("N/A")
        else:
            parts.append(f"{int(v):,}")
    return " / ".join(parts)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _tp_error_header_cells(thresh_str: str) -> list[str]:
    return [f"{tp}@{thresh_str}" for tp in TP_ERROR_TYPES]


def _tp_error_cells(
    label_metrics: dict,
    metric_label: str,
    metric_type: str,
    tp_error_variant: str,
) -> list[str]:
    return [
        _fmt_threshold_vals(
            _get_per_threshold_tp_error(label_metrics, metric_label, metric_type, tp, tp_error_variant)
        )
        for tp in TP_ERROR_TYPES
    ]


def _safe_folder_name(loc_vehicle: str) -> str:
    return loc_vehicle.replace("/", "_")


def _model_ids_from_entries(entries: list[dict]) -> list[str]:
    """Chart-style model ids (name + newline + version), preserving entry order."""
    model_ids: list[str] = []
    for entry in entries:
        mid = f"{entry['model_name']}\n{entry['model_version']}"
        if mid not in model_ids:
            model_ids.append(mid)
    return model_ids


def _chart_model_id_to_display(mid: str) -> str:
    return mid.replace("\n", " ")


def _append_plot_links(lines: list[str], metric_type: str) -> None:
    """Relative markdown image links for generated PNG figures."""
    lines.append("## Figures")
    lines.append("")
    lines.append(f"### AP / mAP / mAPH / NDS (`{metric_type}`)")
    lines.append(f"![AP chart](ap_{metric_type}.png)")
    lines.append("")
    for tp_variant, variant_label in TP_ERROR_CHART_VARIANTS:
        display = _tp_error_variant_display_title(tp_variant, variant_label)
        suffix = f"_{tp_variant}" if tp_variant else ""
        lines.append(f"### Per-class TP error — {display}")
        lines.append(f"![TP error {display}](tp_error_bar_{metric_type}{suffix}.png)")
        lines.append("")
    lines.append(f"### Mean TP error (`{metric_type}`)")
    lines.append(f"![Mean TP error](mean_tp_error_{metric_type}.png)")
    lines.append("")


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


def _dedupe_entries_by_model(entries: list[dict]) -> list[dict]:
    """Keep one entry per (model_name, model_version); first json file in list wins."""
    seen: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for entry in entries:
        key = (entry["model_name"], entry["model_version"])
        if key not in seen:
            order.append(key)
            seen[key] = entry
    return [seen[k] for k in order]


def _group_entries(
    groups: dict[tuple[str, str], list[dict]],
    metric_range: str,
    loc_vehicle: str,
) -> list[dict]:
    return _dedupe_entries_by_model(groups.get((metric_range, loc_vehicle), []))


# ── Markdown report (per location/vehicle) ───────────────────────────────────


def build_location_report(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
) -> str:
    metric_type_display = metric_type.replace("_", " ").title()

    lines: list[str] = []
    lines.append(f"# {loc_vehicle} — {metric_type_display}")
    lines.append("")

    for metric_range in sorted(metric_ranges):
        if (metric_range, loc_vehicle) not in groups:
            continue
        entries = _group_entries(groups, metric_range, loc_vehicle)

        lines.append(f"## {metric_range}")
        lines.append("")

        all_labels: list[str] = []
        for entry in entries:
            idx = _build_metric_label_index(entry)
            for metric_label in idx.keys():
                if metric_label not in all_labels:
                    all_labels.append(metric_label)

        # Build header: label columns show name + GT count
        label_gts: dict[str, int] = {}
        for metric_label in all_labels:
            for entry in entries:
                idx = _build_metric_label_index(entry)
                bucket_label = idx.get(metric_label, (metric_label, {}))[0]
                ml = entry["metadata_label"].get(metric_label) or entry["metadata_label"].get(bucket_label) or {}
                g = ml.get(f"metadata_label/test_{metric_label}_num_ground_truths")
                if g is not None:
                    label_gts[metric_label] = g
                    break

        overall_metric_labels = [label for label, _ in CATEGORICAL_OVERALL_METRICS]
        header_cols = ["Model version", *overall_metric_labels]
        for metric_label in all_labels:
            gts = label_gts.get(metric_label, 0)
            header_cols.append(f"{metric_label}<br>({gts:,})")
        lines.append("| " + " | ".join(header_cols) + " |")

        sep_cols = [":----"] + ["---:"] * len(overall_metric_labels) + ["---:"] * len(all_labels)
        lines.append("| " + " | ".join(sep_cols) + " |")

        # One row per model
        for entry in entries:
            model_id = f"{entry['model_name']} {entry['model_version']}"

            cells = [model_id]
            for name in overall_metric_labels:
                cells.append(_fmt(_overall_metric_value(entry, name, metric_type)))
            idx = _build_metric_label_index(entry)
            for metric_label in all_labels:
                lm = idx.get(metric_label, (metric_label, {}))[1]
                ap = _extract_label_ap(lm, metric_label, metric_type)
                cells.append(_fmt(ap))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        # ── Detailed per-model tables ────────────────────────────────────
        thresholds = THRESHOLDS.get(metric_type, [])
        thresh_str = "/".join(thresholds)

        detail_header = [
            "class_name",
            "GTs",
            _num_match_column_label(None, thresh_str),
            f"AP@{thresh_str}",
            f"max_f1@{thresh_str}",
            f"optimal_conf@{thresh_str}",
        ]
        detail_sep = [":----", "---:", ":----", ":----", ":----", ":----"]

        for entry in entries:
            model_id = f"{entry['model_name']} {entry['model_version']}"
            lines.append(f"**{model_id}**")
            lines.append("")
            lines.append(_markdown_row(detail_header))
            lines.append(_markdown_row(detail_sep))

            idx = _build_metric_label_index(entry)
            for metric_label in all_labels:
                bucket_label, lm = idx.get(metric_label, (metric_label, {}))
                ml = entry["metadata_label"].get(metric_label) or entry["metadata_label"].get(bucket_label) or {}
                gts = _extract_label_gts(ml, metric_label)

                ap_vals = _get_per_threshold(lm, metric_label, metric_type, "AP")
                num_match_vals = _get_per_threshold_num_match(lm, metric_label, metric_type, None)
                f1_vals = _get_per_threshold(lm, metric_label, metric_type, "max-f1score")
                conf_vals = _get_per_threshold(lm, metric_label, metric_type, "optimal-confidence")

                lines.append(
                    _markdown_row(
                        [
                            metric_label,
                            _fmt_int(gts),
                            _fmt_threshold_int_vals(num_match_vals),
                            _fmt_threshold_vals(ap_vals),
                            _fmt_threshold_vals(f1_vals),
                            _fmt_threshold_vals(conf_vals),
                        ]
                    )
                )
            lines.append("")
            _append_model_mean_tp_error_table(lines, entry, metric_type)
            for tp_variant, variant_label in (("", "default"), ("medium", "medium"), ("optimal", "optimal")):
                _append_model_tp_error_variant_table(
                    lines, entry, all_labels, metric_type, tp_variant, variant_label, thresh_str
                )

    _append_plot_links(lines, metric_type)
    return "\n".join(lines)


# ── Bar-chart plot (per location/vehicle) ────────────────────────────────────


def _get_label_gts(
    entries: list[dict],
    metric_label: str,
) -> int:
    """Sum GTs for a metric label across entries (use first model's value since GTs are shared)."""
    for entry in entries:
        idx = _build_metric_label_index(entry)
        bucket_label = idx.get(metric_label, (metric_label, {}))[0]
        ml = entry["metadata_label"].get(metric_label) or entry["metadata_label"].get(bucket_label) or {}
        gts = ml.get(f"metadata_label/test_{metric_label}_num_ground_truths")
        if gts is not None:
            return gts
    return 0


def _get_total_gts(entries: list[dict]) -> int:
    for entry in entries:
        gts = entry["metadata"].get("metadata/test_total_num_ground_truths")
        if gts is not None:
            return gts
    return 0


def _collect_class_labels_from_entries(entries: list[dict]) -> list[str]:
    """Class label order matching the AP chart (first-seen order across entries)."""
    labels: list[str] = []
    for entry in entries:
        idx = _build_metric_label_index(entry)
        for metric_label in idx.keys():
            if metric_label not in labels:
                labels.append(metric_label)
    return labels


def generate_location_plot(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
    out_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    metric_type_short = metric_type.replace("_", " ").title()

    sorted_ranges = sorted(metric_ranges)
    ranges_with_data = [r for r in sorted_ranges if (r, loc_vehicle) in groups]
    if not ranges_with_data:
        return None

    n_subplots = len(ranges_with_data)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 5.5 * n_subplots), squeeze=False)

    for row, metric_range in enumerate(ranges_with_data):
        ax = axes[row, 0]
        entries = _group_entries(groups, metric_range, loc_vehicle)

        all_labels = _collect_class_labels_from_entries(entries)

        label_gts = {lb: _get_label_gts(entries, lb) for lb in all_labels}
        total_gts = _get_total_gts(entries)
        overall_metric_labels = [label for label, _ in CATEGORICAL_OVERALL_METRICS]
        x_tick_labels = [f"{lb}\n(GTs: {label_gts[lb]:,})" for lb in all_labels] + [
            f"mAP\n(GTs: {total_gts:,})" if label == "mAP" else label for label in overall_metric_labels
        ]

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
            for metric_label in all_labels:
                ap_vals = []
                for entry in entries:
                    eid = f"{entry['model_name']}\n{entry['model_version']}"
                    if eid != mid:
                        continue
                    idx = _build_metric_label_index(entry)
                    lm = idx.get(metric_label, (metric_label, {}))[1]
                    ap = _extract_label_ap(lm, metric_label, metric_type)
                    if ap is not None:
                        ap_vals.append(ap)
                values.append(ap_vals[0] if ap_vals else 0.0)

            for name in overall_metric_labels:
                agg_vals = []
                for entry in entries:
                    eid = f"{entry['model_name']}\n{entry['model_version']}"
                    if eid != mid:
                        continue
                    v = _overall_metric_value(entry, name, metric_type)
                    if v is not None:
                        agg_vals.append(v)
                values.append(agg_vals[0] if agg_vals else 0.0)

            offset = (model_idx - (n_models - 1) / 2) * bar_width
            bars = ax.bar(x + offset, values, bar_width, label=mid, zorder=3)
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
        ax.set_xticklabels(x_tick_labels, fontsize=11, rotation=45, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("AP / mAP / mAPH / NDS", fontsize=12)
        ax.set_title(
            f"{metric_range}  ({metric_type_short})",
            fontsize=13,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(loc_vehicle, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    fig.subplots_adjust(hspace=0.22)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ap_{metric_type}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── TP-error stacked bar plots (per location/vehicle) ────────────────────────


def _collect_tp_error_raw_by_model(
    entries: list[dict],
    class_label: str,
    metric_type: str,
    model_ids: list[str],
    tp_error_variant: str = "",
) -> dict[str, list[float | None]]:
    raw_by_model: dict[str, list[float | None]] = {}
    for mid in model_ids:
        raw_vals: list[float | None] = []
        for tp_err in TP_ERROR_TYPES:
            found = None
            for entry in entries:
                eid = f"{entry['model_name']}\n{entry['model_version']}"
                if eid != mid:
                    continue
                idx = _build_metric_label_index(entry)
                lm = idx.get(class_label, (class_label, {}))[1]
                found = _extract_label_tp_error_mean(lm, class_label, metric_type, tp_err, tp_error_variant)
                if found is not None:
                    break
            raw_vals.append(found)
        if any(v is not None for v in raw_vals):
            raw_by_model[mid] = raw_vals
    return raw_by_model


def _plot_tp_error_stacked_bar_panel(
    ax,
    class_labels: list[str],
    data_by_class: dict[str, dict[str, list[float | None]]],
    model_ids: list[str],
    model_cmap,
    segment_colors: list,
) -> bool:
    """Stacked bars grouped by class on x-axis; models distinguished by outline color."""
    import numpy as np

    n_classes = len(class_labels)
    n_models = len(model_ids)
    if n_classes == 0 or n_models == 0:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return False

    x = np.arange(n_classes)
    bar_width = 0.8 / n_models
    panel_max = 0.0
    drew_any = False

    for model_idx, mid in enumerate(model_ids):
        outline_color = model_cmap(model_idx % 10)
        offset = (model_idx - (n_models - 1) / 2) * bar_width
        for class_idx, class_label in enumerate(class_labels):
            raw_by_model = data_by_class.get(class_label, {})
            if mid not in raw_by_model:
                continue
            xpos = x[class_idx] + offset
            bottom = 0.0
            raw_scores = raw_by_model[mid]
            for tp_idx, val in enumerate(raw_scores):
                if val is None:
                    continue
                ax.bar(
                    xpos,
                    val,
                    bar_width,
                    bottom=bottom,
                    color=segment_colors[tp_idx],
                    edgecolor=outline_color,
                    linewidth=2.0,
                    zorder=3,
                )
                ax.text(
                    xpos,
                    bottom + val / 2,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    zorder=4,
                )
                bottom += val
                panel_max = max(panel_max, bottom)
                drew_any = True

    if not drew_any:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        return False

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=13, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, panel_max * 1.15 if panel_max > 0 else 1.0)
    ax.set_ylabel("Stacked mean TP error", fontsize=13)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    return True


def generate_tp_error_bar_plot(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
    out_dir: Path,
    tp_error_variant: str = "",
    variant_label: str = "default",
) -> Path | None:
    """One PNG: one subplot per range bucket; x-axis grouped by class (stacked mean TP errors)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    metric_type_short = metric_type.replace("_", " ").title()

    sorted_ranges = sorted(metric_ranges)
    ranges_with_data = [r for r in sorted_ranges if (r, loc_vehicle) in groups]
    if not ranges_with_data:
        return None

    n_rows = len(ranges_with_data)
    model_cmap = plt.get_cmap("tab10")
    segment_colors = [plt.get_cmap("Set2")(i / len(TP_ERROR_TYPES)) for i in range(len(TP_ERROR_TYPES))]

    display_title = _tp_error_variant_display_title(tp_error_variant, variant_label)
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 6.0 * n_rows), squeeze=False)

    has_any_data = False
    reference_model_ids: list[str] | None = None

    for row, metric_range in enumerate(ranges_with_data):
        ax = axes[row, 0]
        entries = _group_entries(groups, metric_range, loc_vehicle)
        model_ids: list[str] = []
        for entry in entries:
            mid = f"{entry['model_name']}\n{entry['model_version']}"
            if mid not in model_ids:
                model_ids.append(mid)
        if reference_model_ids is None:
            reference_model_ids = model_ids

        class_labels = _collect_class_labels_from_entries(entries)
        if not class_labels:
            ax.set_visible(False)
            continue

        data_by_class = {
            class_label: _collect_tp_error_raw_by_model(entries, class_label, metric_type, model_ids, tp_error_variant)
            for class_label in class_labels
        }
        if _plot_tp_error_stacked_bar_panel(ax, class_labels, data_by_class, model_ids, model_cmap, segment_colors):
            has_any_data = True

        ax.set_title(
            f"{metric_range}  ({metric_type_short})",
            fontsize=15,
            fontweight="bold",
        )

    if not has_any_data:
        plt.close(fig)
        return None

    tp_legend = [Patch(facecolor=segment_colors[i], label=tp) for i, tp in enumerate(TP_ERROR_TYPES)]
    model_legend = [
        Line2D([0], [0], color=model_cmap(i % 10), linewidth=2, label=reference_model_ids[i])
        for i in range(len(reference_model_ids or []))
    ]
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.22)

    tp_leg = fig.legend(
        handles=tp_legend,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(TP_ERROR_TYPES),
        frameon=True,
        title=f"TP error ({display_title})",
        title_fontsize=12,
    )
    bottom_leg = tp_leg
    if model_legend:
        fig.canvas.draw()
        tp_bottom = tp_leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
        model_leg = fig.legend(
            handles=model_legend,
            fontsize=11,
            loc="upper center",
            bbox_to_anchor=(0.5, tp_bottom),
            ncol=min(len(model_legend), 4),
            frameon=True,
            title="Model (outline)",
            title_fontsize=12,
        )
        fig.add_artist(tp_leg)
        fig.add_artist(model_leg)
        bottom_leg = model_leg

    fig.canvas.draw()
    legend_bottom = bottom_leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.subplots_adjust(top=legend_bottom - 0.018, hspace=0.22)

    fig.suptitle(
        f"{loc_vehicle} — TP error {display_title} " f"(mean over thresholds, stacked; lower is better)",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tp_error_variant}" if tp_error_variant else ""
    out_path = out_dir / f"tp_error_bar_{metric_type}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Mean TP-error grouped bar plots (per location/vehicle) ───────────────────


def _mean_tp_error_metric_key(tp_error_variant: str, err_type: str, metric_type: str) -> str:
    if tp_error_variant == "medium":
        return f"T4MetricV2/mean-tp-error-medium-{err_type}_{metric_type}"
    if tp_error_variant == "optimal":
        return f"T4MetricV2/mean-tp-error-optimal-{err_type}_{metric_type}"
    return f"T4MetricV2/mean-tp-error_{err_type}_{metric_type}"


def _mean_tp_error_bucket_name(tp_error_variant: str, err_type: str) -> str:
    if tp_error_variant == "medium":
        return f"mean-tp-error-medium-{err_type}"
    if tp_error_variant == "optimal":
        return f"mean-tp-error-optimal-{err_type}"
    return "mean-tp-error"


def _collect_mean_tp_error_by_model(
    entries: list[dict],
    metric_type: str,
    model_ids: list[str],
    tp_error_variant: str,
) -> dict[str, list[float | None]]:
    raw_by_model: dict[str, list[float | None]] = {}
    for mid in model_ids:
        raw_vals: list[float | None] = []
        for err_type in MEAN_TP_ERROR_TYPES:
            found = None
            for entry in entries:
                eid = f"{entry['model_name']}\n{entry['model_version']}"
                if eid != mid:
                    continue
                bucket_name = _mean_tp_error_bucket_name(tp_error_variant, err_type)
                bucket = entry.get("label_metrics", {}).get(bucket_name, {})
                key = _mean_tp_error_metric_key(tp_error_variant, err_type, metric_type)
                val = bucket.get(key)
                if val is not None:
                    found = val
                    break
            raw_vals.append(found)
        if any(v is not None for v in raw_vals):
            raw_by_model[mid] = raw_vals
    return raw_by_model


def _append_model_tp_error_variant_table(
    lines: list[str],
    entry: dict,
    all_labels: list[str],
    metric_type: str,
    tp_error_variant: str,
    variant_label: str,
    thresh_str: str,
) -> None:
    """Per-class TP error table for medium or optimal variant."""
    display = _tp_error_variant_display_title(tp_error_variant, variant_label)
    lines.append(f"**TP error — {display}**")
    lines.append("")
    num_match_col = _num_match_column_label(tp_error_variant, thresh_str)
    lines.append(_markdown_row(["class_name", num_match_col, *_tp_error_header_cells(thresh_str)]))
    lines.append(_markdown_row([":----", ":----", *[":----"] * len(TP_ERROR_TYPES)]))

    idx = _build_metric_label_index(entry)
    for metric_label in all_labels:
        lm = idx.get(metric_label, (metric_label, {}))[1]
        num_match_vals = _get_per_threshold_num_match(lm, metric_label, metric_type, tp_error_variant)
        lines.append(
            _markdown_row(
                [
                    metric_label,
                    _fmt_threshold_int_vals(num_match_vals),
                    *_tp_error_cells(lm, metric_label, metric_type, tp_error_variant),
                ]
            )
        )
    lines.append("")


def _append_model_mean_tp_error_table(lines: list[str], entry: dict, metric_type: str) -> None:
    """Mean TP error table (default / medium / optimal) below one model's class detail table."""
    mid = f"{entry['model_name']}\n{entry['model_version']}"
    lines.append("**Mean TP error**")
    lines.append("")
    lines.append(_markdown_row(["Variant", *MEAN_TP_ERROR_TYPES]))
    lines.append(_markdown_row([":----", *["---:"] * len(MEAN_TP_ERROR_TYPES)]))
    for tp_variant, variant_label in TP_ERROR_CHART_VARIANTS:
        display = _tp_error_variant_display_title(tp_variant, variant_label)
        raw_by_model = _collect_mean_tp_error_by_model([entry], metric_type, [mid], tp_variant)
        if mid in raw_by_model:
            vals = [_fmt(v) for v in raw_by_model[mid]]
        else:
            vals = ["N/A"] * len(MEAN_TP_ERROR_TYPES)
        lines.append(_markdown_row([display, *vals]))
    lines.append("")


def _plot_mean_tp_error_bar_cell(
    ax,
    raw_by_model: dict[str, list[float | None]],
    model_ids: list[str],
    model_cmap,
    *,
    show_ylabel: bool = True,
) -> bool:
    """Grouped bars on x-axis by mean TP error name (mATE … mAAE)."""
    import numpy as np

    models_present = [mid for mid in model_ids if mid in raw_by_model]
    if not models_present:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", fontsize=11)
        return False

    x = np.arange(len(MEAN_TP_ERROR_TYPES))
    n_models = len(models_present)
    bar_width = 0.8 / n_models
    panel_max = 0.0
    drew_any = False

    for model_idx, mid in enumerate(models_present):
        raw_scores = raw_by_model[mid]
        offset = (model_idx - (n_models - 1) / 2) * bar_width
        color = model_cmap(model_ids.index(mid) % 10)
        heights = [0.0 if v is None else v for v in raw_scores]
        bars = ax.bar(
            x + offset,
            heights,
            bar_width,
            label=mid,
            color=color,
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )
        for bar, val in zip(bars, raw_scores):
            if val is None:
                continue
            panel_max = max(panel_max, val)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
                zorder=4,
            )
        drew_any = True

    ax.set_xticks(x)
    ax.set_xticklabels(MEAN_TP_ERROR_TYPES, fontsize=12, fontweight="bold", rotation=45, ha="right")
    ax.set_ylim(0, panel_max * 1.18 if panel_max > 0 else 1.0)
    if show_ylabel:
        ax.set_ylabel("Mean TP error", fontsize=11)
    ax.grid(axis="y", linestyle="-", linewidth=0.8, color="#b0b0b0", alpha=1.0, zorder=0)
    return drew_any


def generate_mean_tp_error_bar_plot(
    loc_vehicle: str,
    groups: dict[tuple[str, str], list[dict]],
    metric_ranges: list[str],
    metric_type: str,
    out_dir: Path,
) -> Path | None:
    """Bar chart: rows = range buckets; columns = default, medium, optimal; x-axis = mean TP errors."""
    import matplotlib.pyplot as plt

    metric_type_short = metric_type.replace("_", " ").title()
    sorted_ranges = sorted(metric_ranges)
    ranges_with_data = [r for r in sorted_ranges if (r, loc_vehicle) in groups]
    if not ranges_with_data:
        return None

    n_rows = len(ranges_with_data)
    n_cols = len(TP_ERROR_CHART_VARIANTS)
    model_cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols + 1.5, 5.0 * n_rows + 2.0),
        squeeze=False,
    )

    has_any_data = False
    legend_handles = None
    legend_labels = None

    for row, metric_range in enumerate(ranges_with_data):
        entries = _group_entries(groups, metric_range, loc_vehicle)
        model_ids: list[str] = []
        for entry in entries:
            mid = f"{entry['model_name']}\n{entry['model_version']}"
            if mid not in model_ids:
                model_ids.append(mid)

        for col, (tp_variant, variant_label) in enumerate(TP_ERROR_CHART_VARIANTS):
            ax = axes[row, col]
            raw_by_model = _collect_mean_tp_error_by_model(entries, metric_type, model_ids, tp_variant)
            if _plot_mean_tp_error_bar_cell(ax, raw_by_model, model_ids, model_cmap, show_ylabel=(col != 0)):
                has_any_data = True
                if legend_handles is None and raw_by_model:
                    legend_handles, legend_labels = ax.get_legend_handles_labels()

            if row == 0:
                ax.set_title(
                    _mean_tp_column_title(tp_variant, variant_label),
                    fontsize=20,
                    fontweight="bold",
                    pad=28,
                )

        axes[row, 0].text(
            -0.16,
            0.5,
            f"{metric_type}\n{metric_range}",
            transform=axes[row, 0].transAxes,
            fontsize=13,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            clip_on=False,
        )

    if not has_any_data:
        plt.close(fig)
        return None

    fig.tight_layout()
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.12, hspace=0.18, wspace=0.35)

    suptitle_artist = fig.suptitle(
        f"{loc_vehicle} — mean TP error ({metric_type_short})\n"
        f"default (recall @0.10) / medium (recall @0.40) / optimal — lower is better",
        fontsize=17,
        fontweight="bold",
        y=1.0,
    )

    if legend_handles:
        fig.canvas.draw()
        title_bottom = suptitle_artist.get_window_extent().transformed(fig.transFigure.inverted()).y0
        model_leg = fig.legend(
            legend_handles,
            legend_labels,
            fontsize=13,
            loc="upper center",
            bbox_to_anchor=(0.5, title_bottom),
            ncol=min(len(legend_labels), 4),
            frameon=True,
            title="Model",
            title_fontsize=14,
        )
        fig.canvas.draw()
        legend_bottom = model_leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
        fig.subplots_adjust(
            left=0.22,
            right=0.98,
            top=legend_bottom - 0.055,
            bottom=0.12,
            hspace=0.18,
            wspace=0.35,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mean_tp_error_{metric_type}.png"
    fig.savefig(out_path, dpi=175, bbox_inches="tight", pad_inches=0.25)
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

    all_loc_vehicles: list[str] = sorted({lv for lvs in loc_vehicles_seen.values() for lv in lvs})

    if args.output_dir:
        for loc_vehicle in all_loc_vehicles:
            sub_dir = args.output_dir / _safe_folder_name(loc_vehicle)
            sub_dir.mkdir(parents=True, exist_ok=True)

            for mt in metric_types:
                report = build_location_report(loc_vehicle, groups, metric_ranges_seen, mt)
                md_path = sub_dir / f"report_{mt}.md"
                md_path.write_text(report)
                print(f"  {md_path}")

                png_path = generate_location_plot(loc_vehicle, groups, metric_ranges_seen, mt, sub_dir)
                if png_path:
                    print(f"  {png_path}")

                for tp_variant, tp_variant_label in TP_ERROR_CHART_VARIANTS:
                    tp_bar_path = generate_tp_error_bar_plot(
                        loc_vehicle,
                        groups,
                        metric_ranges_seen,
                        mt,
                        sub_dir,
                        tp_error_variant=tp_variant,
                        variant_label=tp_variant_label,
                    )
                    if tp_bar_path:
                        print(f"  {tp_bar_path}")

                mean_tp_path = generate_mean_tp_error_bar_plot(loc_vehicle, groups, metric_ranges_seen, mt, sub_dir)
                if mean_tp_path:
                    print(f"  {mean_tp_path}")
    else:
        for mt in metric_types:
            for loc_vehicle in all_loc_vehicles:
                report = build_location_report(loc_vehicle, groups, metric_ranges_seen, mt)
                print(report)


if __name__ == "__main__":
    main()
