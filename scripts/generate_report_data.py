#!/usr/bin/env python3
"""Generate RMSE datasets for the CT project report."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from skimage import io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ct_app.app_config import (
    EXPERIMENT_DETECTORS,
    EXPERIMENT_FAN_SPAN,
    EXPERIMENT_PARALLEL_SPAN,
    EXPERIMENT_SCANS,
    GEOMETRY_FAN,
    GEOMETRY_PARALLEL,
)
from ct_app.experiment_data import get_experiment_options
from ct_app.image_utils import (
    calculate_rmse,
    filter_sinogram,
    list_sample_images,
    preprocess_image,
    resolve_sample_dir,
    stabilize_sinogram,
)
from ct_app.reconstruction import (
    compute_selected_geometry_indices,
    iradon_transform,
    radon_transform,
)

DEFAULT_FILTER_COMPARE_IMAGES = ["Shepp_logan.jpg", "CT_ScoutView.jpg"]

FULL_RANGES: Dict[str, List[int]] = {
    EXPERIMENT_DETECTORS: list(range(90, 721, 90)),
    EXPERIMENT_SCANS: list(range(90, 721, 90)),
    EXPERIMENT_FAN_SPAN: list(range(45, 271, 45)),
    EXPERIMENT_PARALLEL_SPAN: list(range(50, 201, 25)),
}


def slugify(value: str) -> str:
    polish_map = str.maketrans({
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    })
    value = value.translate(polish_map)
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value.lower()).strip("_")
    return slug or "data"


def get_ranges(quick: bool) -> Dict[str, List[int]]:
    if quick:
        print("[INFO] Quick mode keeps full parameter ranges to preserve report consistency.")
    return FULL_RANGES


def load_input_image(image_name: str, image_path: Optional[str]) -> Tuple[np.ndarray, str]:
    if image_path:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        raw_image = io.imread(path)
        return preprocess_image(raw_image), str(path)

    sample_dir = resolve_sample_dir()
    available_images = list_sample_images(sample_dir)
    if image_name not in available_images:
        raise FileNotFoundError(
            f"Sample image '{image_name}' not found. Available: {available_images}"
        )

    sample_path = sample_dir / image_name
    raw_image = io.imread(sample_path)
    return preprocess_image(raw_image), str(sample_path)


def run_param_sweep(
    input_image: np.ndarray,
    beam_geometry: str,
    experiment_type: str,
    parameter_values: Iterable[int],
    radius: float,
    width: int,
    height: int,
    fan_span_rad: float,
    parallel_span_scale: float,
) -> Tuple[List[int], List[float]]:
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius_corr = float(np.sqrt(center_x**2 + center_y**2))

    values = [int(v) for v in parameter_values]
    rmse_values: List[float] = []

    for value in values:
        scan_steps = 180
        detector_count = 180
        fan_span_current = fan_span_rad
        parallel_span_current = parallel_span_scale

        if experiment_type == EXPERIMENT_DETECTORS:
            detector_count = value
        elif experiment_type == EXPERIMENT_SCANS:
            scan_steps = value
        elif experiment_type == EXPERIMENT_FAN_SPAN:
            fan_span_current = float(np.radians(value))
        elif experiment_type == EXPERIMENT_PARALLEL_SPAN:
            parallel_span_current = value / 100.0
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
            beam_geometry,
            scan_steps,
            radius_corr,
            detector_count,
            fan_span_current,
            parallel_span_current,
            center_x,
            center_y,
        )

        sinogram = stabilize_sinogram(radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx))
        filtered_sinogram = filter_sinogram(sinogram).astype(np.float32)
        reconstruction = iradon_transform(
            filtered_sinogram,
            xe_idx,
            ye_idx,
            xd_idx,
            yd_idx,
            height,
            width,
        )
        rmse_values.append(float(calculate_rmse(input_image, reconstruction)))

    return values, rmse_values


def reconstruct_image(
    input_image: np.ndarray,
    beam_geometry: str,
    scan_steps: int,
    detector_count: int,
    fan_span_rad: float,
    parallel_span_scale: float,
    use_filter: bool,
) -> np.ndarray:
    height, width = input_image.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = float(np.sqrt(center_x**2 + center_y**2))

    xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
        beam_geometry,
        scan_steps,
        radius,
        detector_count,
        fan_span_rad,
        parallel_span_scale,
        center_x,
        center_y,
    )

    sinogram = stabilize_sinogram(radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx))
    if use_filter:
        sinogram = filter_sinogram(sinogram).astype(np.float32)

    return iradon_transform(
        sinogram,
        xe_idx,
        ye_idx,
        xd_idx,
        yd_idx,
        height,
        width,
    )


def write_generic_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["geometry", "experiment", "parameter", "rmse"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate RMSE datasets for CT report.")
    parser.add_argument(
        "--image",
        default="CT_ScoutView.jpg",
        help="Sample image name from tomograf-obrazy.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Path to external image (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default="report_data",
        help="Output directory for generated datasets.",
    )
    parser.add_argument(
        "--geometry",
        default="all",
        choices=["all", GEOMETRY_FAN, GEOMETRY_PARALLEL],
        help="Beam geometry to process.",
    )
    parser.add_argument(
        "--fan-span-deg",
        type=float,
        default=180.0,
        help="Base fan span in degrees.",
    )
    parser.add_argument(
        "--parallel-span-pct",
        type=float,
        default=100.0,
        help="Base parallel detector span in percent of diagonal.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Compatibility flag; keeps full parameter ranges for consistent report data.",
    )
    parser.add_argument(
        "--skip-filter-comparison",
        action="store_true",
        help="Skip RMSE comparison with and without filtering.",
    )
    parser.add_argument(
        "--filter-compare-images",
        nargs="+",
        default=None,
        help="Sample image names used for filter-vs-no-filter comparison.",
    )
    parser.add_argument(
        "--filter-compare-geometry",
        default=GEOMETRY_FAN,
        choices=[GEOMETRY_FAN, GEOMETRY_PARALLEL],
        help="Geometry used in filter-vs-no-filter comparison.",
    )
    parser.add_argument(
        "--filter-compare-detectors",
        type=int,
        default=360,
        help="Detector count used in filter-vs-no-filter comparison.",
    )
    parser.add_argument(
        "--filter-compare-scans",
        type=int,
        default=360,
        help="Scan steps used in filter-vs-no-filter comparison.",
    )
    parser.add_argument(
        "--filter-compare-fan-span-deg",
        type=float,
        default=270.0,
        help="Fan span in degrees used in filter-vs-no-filter comparison.",
    )
    parser.add_argument(
        "--filter-compare-parallel-span-pct",
        type=float,
        default=100.0,
        help="Parallel span in percent used in filter-vs-no-filter comparison.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_image, input_source = load_input_image(args.image, args.image_path)
    height, width = input_image.shape
    radius = float(np.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2))
    fan_span_rad = float(np.radians(args.fan_span_deg))
    parallel_span_scale = float(args.parallel_span_pct / 100.0)

    geometries = [GEOMETRY_FAN, GEOMETRY_PARALLEL] if args.geometry == "all" else [args.geometry]
    ranges = get_ranges(args.quick)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).expanduser().resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for geometry in geometries:
        print(f"[INFO] Geometry: {geometry}")
        for experiment_type in get_experiment_options(geometry):
            parameter_values = ranges[experiment_type]
            print(f"[INFO]   Experiment: {experiment_type} | points: {len(parameter_values)}")

            params, rmse_values = run_param_sweep(
                input_image=input_image,
                beam_geometry=geometry,
                experiment_type=experiment_type,
                parameter_values=parameter_values,
                radius=radius,
                width=width,
                height=height,
                fan_span_rad=fan_span_rad,
                parallel_span_scale=parallel_span_scale,
            )

            experiment_rows: List[Dict[str, object]] = []
            for param, rmse in zip(params, rmse_values):
                row = {
                    "geometry": geometry,
                    "experiment": experiment_type,
                    "parameter": int(param),
                    "rmse": float(rmse),
                }
                combined_rows.append(row)
                experiment_rows.append(row)

            experiment_csv = run_dir / f"rmse_{slugify(geometry)}_{slugify(experiment_type)}.csv"
            write_csv(experiment_csv, experiment_rows)

            best_idx = int(np.argmin(rmse_values))
            summary_rows.append(
                {
                    "geometry": geometry,
                    "experiment": experiment_type,
                    "points": len(params),
                    "best_parameter": int(params[best_idx]),
                    "best_rmse": float(rmse_values[best_idx]),
                    "min_rmse": float(np.min(rmse_values)),
                    "max_rmse": float(np.max(rmse_values)),
                    "mean_rmse": float(np.mean(rmse_values)),
                    "csv_file": experiment_csv.name,
                }
            )

    combined_csv = run_dir / "report_data.csv"
    write_csv(combined_csv, combined_rows)

    filter_comparison_rows: List[Dict[str, object]] = []
    filter_comparison_csv_name = None
    if not args.skip_filter_comparison:
        filter_image_names = args.filter_compare_images or DEFAULT_FILTER_COMPARE_IMAGES
        filter_geometry = args.filter_compare_geometry
        filter_fan_span_rad = float(np.radians(args.filter_compare_fan_span_deg))
        filter_parallel_span_scale = float(args.filter_compare_parallel_span_pct / 100.0)

        print(f"[INFO] Filter comparison geometry: {filter_geometry}")
        for image_name in filter_image_names:
            filter_image, filter_source = load_input_image(image_name, None)

            reconstruction_without_filter = reconstruct_image(
                filter_image,
                filter_geometry,
                args.filter_compare_scans,
                args.filter_compare_detectors,
                filter_fan_span_rad,
                filter_parallel_span_scale,
                use_filter=False,
            )
            reconstruction_with_filter = reconstruct_image(
                filter_image,
                filter_geometry,
                args.filter_compare_scans,
                args.filter_compare_detectors,
                filter_fan_span_rad,
                filter_parallel_span_scale,
                use_filter=True,
            )

            rmse_without_filter = float(calculate_rmse(filter_image, reconstruction_without_filter))
            rmse_with_filter = float(calculate_rmse(filter_image, reconstruction_with_filter))
            filter_comparison_rows.append(
                {
                    "image": image_name,
                    "image_source": filter_source,
                    "geometry": filter_geometry,
                    "detectors": int(args.filter_compare_detectors),
                    "scans": int(args.filter_compare_scans),
                    "fan_span_deg": float(args.filter_compare_fan_span_deg),
                    "parallel_span_pct": float(args.filter_compare_parallel_span_pct),
                    "rmse_without_filter": rmse_without_filter,
                    "rmse_with_filter": rmse_with_filter,
                    "delta_rmse": rmse_without_filter - rmse_with_filter,
                }
            )

        filter_comparison_csv = run_dir / "filter_comparison.csv"
        write_generic_csv(
            filter_comparison_csv,
            filter_comparison_rows,
            [
                "image",
                "image_source",
                "geometry",
                "detectors",
                "scans",
                "fan_span_deg",
                "parallel_span_pct",
                "rmse_without_filter",
                "rmse_with_filter",
                "delta_rmse",
            ],
        )
        filter_comparison_csv_name = filter_comparison_csv.name

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": input_source,
        "image_shape": [int(height), int(width)],
        "fan_span_deg": float(args.fan_span_deg),
        "parallel_span_pct": float(args.parallel_span_pct),
        "quick_mode": bool(args.quick),
        "geometries": geometries,
        "experiments": summary_rows,
        "combined_csv": combined_csv.name,
        "filter_comparison_csv": filter_comparison_csv_name,
        "filter_comparison": filter_comparison_rows,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"[OK] Saved report data in: {run_dir}")
    print(f"[OK] Combined CSV: {combined_csv}")
    print(f"[OK] Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
