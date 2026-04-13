import numpy as np

from .app_config import (
    EXPERIMENT_DETECTORS,
    EXPERIMENT_FAN_SPAN,
    EXPERIMENT_PARALLEL_SPAN,
    EXPERIMENT_SCANS,
    GEOMETRY_FAN,
)
from .image_utils import calculate_rmse, filter_sinogram
from .reconstruction import compute_selected_geometry_indices, iradon_transform, radon_transform


def get_experiment_options(beam_geometry):
    """Zwraca listę eksperymentów dostępną dla wybranej geometrii."""
    if beam_geometry == GEOMETRY_FAN:
        return [EXPERIMENT_DETECTORS, EXPERIMENT_SCANS, EXPERIMENT_FAN_SPAN]
    return [EXPERIMENT_DETECTORS, EXPERIMENT_SCANS, EXPERIMENT_PARALLEL_SPAN]


def run_experiment(
    input_image,
    beam_geometry,
    experiment_type,
    radius,
    width,
    height,
    fan_span_rad,
    parallel_span_scale,
):
    """Uruchamia serię obliczeń RMSE dla wybranego eksperymentu."""
    parameter_range = []
    rmse_values = []

    if experiment_type == EXPERIMENT_DETECTORS:
        parameter_range = range(90, 721, 90)
        for detector_count_exp in parameter_range:
            xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
                beam_geometry,
                180,
                radius,
                detector_count_exp,
                fan_span_rad,
                parallel_span_scale,
                width / 2.0,
                height / 2.0,
            )
            sinogram_exp = radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx)
            sinogram_exp = filter_sinogram(sinogram_exp).astype(np.float32)
            reconstruction_exp = iradon_transform(sinogram_exp, xe_idx, ye_idx, xd_idx, yd_idx, height, width)
            rmse_values.append(calculate_rmse(input_image, reconstruction_exp))

    elif experiment_type == EXPERIMENT_SCANS:
        parameter_range = range(90, 721, 90)
        for scan_steps_exp in parameter_range:
            xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
                beam_geometry,
                scan_steps_exp,
                radius,
                180,
                fan_span_rad,
                parallel_span_scale,
                width / 2.0,
                height / 2.0,
            )
            sinogram_exp = radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx)
            sinogram_exp = filter_sinogram(sinogram_exp).astype(np.float32)
            reconstruction_exp = iradon_transform(sinogram_exp, xe_idx, ye_idx, xd_idx, yd_idx, height, width)
            rmse_values.append(calculate_rmse(input_image, reconstruction_exp))

    elif experiment_type == EXPERIMENT_FAN_SPAN:
        parameter_range = range(45, 271, 45)
        for fan_span_deg_exp in parameter_range:
            fan_span_rad_exp = np.radians(fan_span_deg_exp)
            xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
                beam_geometry,
                180,
                radius,
                180,
                fan_span_rad_exp,
                parallel_span_scale,
                width / 2.0,
                height / 2.0,
            )
            sinogram_exp = radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx)
            sinogram_exp = filter_sinogram(sinogram_exp).astype(np.float32)
            reconstruction_exp = iradon_transform(sinogram_exp, xe_idx, ye_idx, xd_idx, yd_idx, height, width)
            rmse_values.append(calculate_rmse(input_image, reconstruction_exp))

    elif experiment_type == EXPERIMENT_PARALLEL_SPAN:
        parameter_range = range(50, 201, 25)
        for parallel_span_pct_exp in parameter_range:
            parallel_span_scale_exp = parallel_span_pct_exp / 100.0
            xe_idx, ye_idx, xd_idx, yd_idx = compute_selected_geometry_indices(
                beam_geometry,
                180,
                radius,
                180,
                fan_span_rad,
                parallel_span_scale_exp,
                width / 2.0,
                height / 2.0,
            )
            sinogram_exp = radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx)
            sinogram_exp = filter_sinogram(sinogram_exp).astype(np.float32)
            reconstruction_exp = iradon_transform(sinogram_exp, xe_idx, ye_idx, xd_idx, yd_idx, height, width)
            rmse_values.append(calculate_rmse(input_image, reconstruction_exp))

    return list(parameter_range), rmse_values
