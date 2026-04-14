import numpy as np
import pydicom
from skimage import io
from skimage import transform

from .app_config import SOURCE_BUILTIN
from .image_utils import filter_sinogram, preprocess_image
from .reconstruction import (
    compute_selected_geometry_indices,
    iradon_transform_with_history,
    radon_transform,
)


def load_input_image(source_mode, selected_sample, uploaded_file, sample_dir):
    """Wczytuje i normalizuje obraz z wybranego źródła."""
    input_image = None
    input_identifier = None
    input_label = None
    input_study_date = None

    try:
        if source_mode == SOURCE_BUILTIN and selected_sample is not None:
            sample_path = sample_dir / selected_sample
            input_image = preprocess_image(io.imread(sample_path))
            input_identifier = f"sample::{sample_path}"
            input_label = selected_sample
        elif uploaded_file is not None:
            upload_name = uploaded_file.name.lower()
            if upload_name.endswith(".dcm"):
                dataset = pydicom.dcmread(uploaded_file)
                pixel_data = dataset.pixel_array.astype(np.float32)
                slope = float(getattr(dataset, "RescaleSlope", 1.0))
                intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
                pixel_data = np.squeeze(pixel_data * slope + intercept)
                input_image = preprocess_image(pixel_data)

                study_date_raw = getattr(dataset, "StudyDate", "")
                study_date_digits = "".join(ch for ch in str(study_date_raw) if ch.isdigit())
                if len(study_date_digits) == 8:
                    input_study_date = study_date_digits
            else:
                input_image = preprocess_image(io.imread(uploaded_file))

            input_identifier = f"upload::{uploaded_file.name}:{uploaded_file.size}"
            input_label = uploaded_file.name
    except Exception as err:
        return None, None, None, None, str(err)

    return input_image, input_identifier, input_label, input_study_date, None


def build_result_signature(
    recon_algo_version,
    input_identifier,
    beam_geometry,
    scan_steps,
    detector_count,
    fan_span_deg,
    parallel_span_pct,
    use_filter,
):
    """Tworzy podpis konfiguracji do walidacji cache wyników."""
    return (
        recon_algo_version,
        input_identifier,
        beam_geometry,
        scan_steps,
        detector_count,
        fan_span_deg,
        parallel_span_pct,
        use_filter,
    )


def run_simulation(
    input_image,
    beam_geometry,
    scan_steps,
    detector_count,
    fan_span_rad,
    parallel_span_scale,
    use_filter,
):
    """Wykonuje pełną symulację i zwraca komplet danych do prezentacji."""
    height, width = input_image.shape
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius = np.sqrt(center_x**2 + center_y**2)

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

    sinogram = radon_transform(input_image, xe_idx, ye_idx, xd_idx, yd_idx)

    if use_filter:
        sinogram_to_recon = filter_sinogram(sinogram).astype(np.float32)
    else:
        sinogram_to_recon = sinogram

    reconstruction, reconstruction_history, hit_count_map = iradon_transform_with_history(
        sinogram_to_recon,
        xe_idx,
        ye_idx,
        xd_idx,
        yd_idx,
        height,
        width,
    )

    return {
        "sinogram_data": sinogram,
        "reconstruction_history": reconstruction_history,
        "hit_count_map": hit_count_map,
        "final_reconstruction": reconstruction,
        "width": width,
        "height": height,
        "radius": radius,
    }


def has_matching_result(session_state, current_signature):
    """Sprawdza, czy wyniki w sesji pasują do bieżącej konfiguracji."""
    return (
        "sinogram_data" in session_state
        and "reconstruction_history" in session_state
        and session_state.get("result_signature") == current_signature
    )


def save_simulation_result(session_state, sim_result, current_signature):
    """Zapisuje wyniki symulacji do stanu sesji Streamlit."""
    session_state["sinogram_data"] = sim_result["sinogram_data"]
    session_state["reconstruction_history"] = sim_result["reconstruction_history"]
    session_state["hit_count_map"] = sim_result["hit_count_map"]
    session_state["final_reconstruction"] = sim_result["final_reconstruction"]
    session_state["result_signature"] = current_signature


def _normalize_for_display(image):
    max_val = float(np.max(image))
    if max_val > 0.0:
        return image / max_val
    return image


def _resize_sinogram_for_display(sinogram, target_size=256):
    """Skaluje sinogram do stalego rozmiaru podgladu (UI), bez zmiany danych obliczeniowych."""
    if sinogram.ndim != 2:
        return sinogram

    resized = transform.resize(
        sinogram,
        (target_size, target_size),
        anti_aliasing=True,
        preserve_range=True,
        mode="reflect",
    )
    return resized.astype(np.float32)


def build_preview_frames(sinogram_data, reconstruction_history, hit_count_map, step_idx):
    """Buduje podgląd sinogramu i rekonstrukcji dla bieżącego kroku."""
    max_steps = reconstruction_history.shape[0]

    current_sin = sinogram_data.copy()
    if step_idx + 1 < current_sin.shape[0]:
        current_sin[step_idx + 1 :, :] = 0.0

    current_rec = reconstruction_history[step_idx].copy()
    if hit_count_map is not None:
        progress = (step_idx + 1) / max_steps
        coverage = hit_count_map * progress
        current_rec = np.divide(
            current_rec,
            np.maximum(coverage, 1.0),
            out=np.zeros_like(current_rec),
        )

    sinogram_preview = _resize_sinogram_for_display(current_sin)
    return _normalize_for_display(sinogram_preview), _normalize_for_display(current_rec)


def build_snapshot_frames(reconstruction_history, hit_count_map, snapshot_count):
    """Tworzy obrazy kilku kroków pośrednich rekonstrukcji."""
    max_steps = reconstruction_history.shape[0]
    snapshot_indices = np.linspace(0, max_steps - 1, snapshot_count, dtype=np.int32)

    snapshot_images = []
    for idx in snapshot_indices:
        snapshot = reconstruction_history[int(idx)].copy()
        if hit_count_map is not None:
            snap_progress = (int(idx) + 1) / max_steps
            snap_coverage = hit_count_map * snap_progress
            snapshot = np.divide(
                snapshot,
                np.maximum(snap_coverage, 1.0),
                out=np.zeros_like(snapshot),
            )
        snapshot_images.append(_normalize_for_display(snapshot))

    return snapshot_indices, snapshot_images
