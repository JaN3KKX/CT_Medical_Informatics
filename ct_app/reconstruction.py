import numpy as np
from numba import njit


@njit(cache=True)
def compute_fan_geometry_indices(steps, r, n, phi, cx, cy):
    """Prekalkuluje indeksy zrodla i detektorow dla geometrii wachlarzowej."""
    xe_idx = np.empty((steps, n), dtype=np.int32)
    ye_idx = np.empty((steps, n), dtype=np.int32)
    xd_idx = np.empty((steps, n), dtype=np.int32)
    yd_idx = np.empty((steps, n), dtype=np.int32)

    det_step = phi / (n - 1) if n > 1 else 0.0

    for s in range(steps):
        alpha = s * (2.0 * np.pi / steps)
        xe = r * np.cos(alpha)
        ye = r * np.sin(alpha)
        xe_scalar = int(np.rint(xe + cx))
        ye_scalar = int(np.rint(ye + cy))

        angle0 = alpha + np.pi - phi / 2.0
        for i in range(n):
            xe_idx[s, i] = xe_scalar
            ye_idx[s, i] = ye_scalar
            angle = angle0 + i * det_step
            xd_idx[s, i] = int(np.rint(r * np.cos(angle) + cx))
            yd_idx[s, i] = int(np.rint(r * np.sin(angle) + cy))

    return xe_idx, ye_idx, xd_idx, yd_idx


@njit(cache=True)
def compute_parallel_geometry_indices(steps, r, n, span_scale, cx, cy):
    """Prekalkuluje konce promieni dla geometrii rownoleglej."""
    x0_idx = np.empty((steps, n), dtype=np.int32)
    y0_idx = np.empty((steps, n), dtype=np.int32)
    x1_idx = np.empty((steps, n), dtype=np.int32)
    y1_idx = np.empty((steps, n), dtype=np.int32)

    offset_max = r * span_scale
    offset_step = (2.0 * offset_max) / (n - 1) if n > 1 else 0.0
    half_length = 2.0 * r

    for s in range(steps):
        alpha = s * (2.0 * np.pi / steps)
        dir_x = np.cos(alpha)
        dir_y = np.sin(alpha)
        norm_x = -dir_y
        norm_y = dir_x

        for i in range(n):
            offset = -offset_max + i * offset_step
            base_x = offset * norm_x
            base_y = offset * norm_y

            x0 = base_x - half_length * dir_x
            y0 = base_y - half_length * dir_y
            x1 = base_x + half_length * dir_x
            y1 = base_y + half_length * dir_y

            x0_idx[s, i] = int(np.rint(x0 + cx))
            y0_idx[s, i] = int(np.rint(y0 + cy))
            x1_idx[s, i] = int(np.rint(x1 + cx))
            y1_idx[s, i] = int(np.rint(y1 + cy))

    return x0_idx, y0_idx, x1_idx, y1_idx


def compute_selected_geometry_indices(
    beam_geometry,
    steps,
    radius,
    detector_count,
    fan_span_rad,
    parallel_span_scale,
    cx,
    cy,
):
    """Wybiera sposob generowania geometrii na podstawie wybranego modelu wiazki."""
    if beam_geometry == "Równoległa":
        return compute_parallel_geometry_indices(steps, radius, detector_count, parallel_span_scale, cx, cy)
    return compute_fan_geometry_indices(steps, radius, detector_count, fan_span_rad, cx, cy)


@njit(cache=True)
def line_integral(image, x0, y0, x1, y1):
    """Sumuje wartosci pikseli i liczy poprawne probki na odcinku."""
    h, w = image.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    x = x0
    y = y0

    line_sum = 0.0
    valid_samples = 0

    if dx > dy:
        err = dx / 2.0
        while True:
            if 0 <= x < w and 0 <= y < h:
                line_sum += image[y, x]
                valid_samples += 1
            if x == x1:
                break
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while True:
            if 0 <= x < w and 0 <= y < h:
                line_sum += image[y, x]
                valid_samples += 1
            if y == y1:
                break
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    return line_sum, valid_samples


@njit(cache=True)
def backproject_line(output_image, hit_count, x0, y0, x1, y1, val):
    """Dodaje wartosc projekcji na odcinku i sledzi pokrycie pikseli."""
    h, w = output_image.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    x = x0
    y = y0

    if dx > dy:
        err = dx / 2.0
        while True:
            if 0 <= x < w and 0 <= y < h:
                output_image[y, x] += val
                hit_count[y, x] += 1.0
            if x == x1:
                break
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while True:
            if 0 <= x < w and 0 <= y < h:
                output_image[y, x] += val
                hit_count[y, x] += 1.0
            if y == y1:
                break
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy


@njit(cache=True)
def radon_transform(image, xe_idx, ye_idx, xd_idx, yd_idx):
    """Liczy caly sinogram w jednym wywolaniu Numba."""
    steps = xe_idx.shape[0]
    n = xe_idx.shape[1]
    sinogram = np.zeros((steps, n), dtype=np.float32)

    for s in range(steps):
        for i in range(n):
            x0 = xe_idx[s, i]
            y0 = ye_idx[s, i]
            x1 = xd_idx[s, i]
            y1 = yd_idx[s, i]
            line_sum, valid_samples = line_integral(image, x0, y0, x1, y1)
            if valid_samples > 0:
                sinogram[s, i] = line_sum / valid_samples

    return sinogram


@njit(cache=True)
def iradon_transform(sinogram, xe_idx, ye_idx, xd_idx, yd_idx, h, w):
    """Rekonstrukcja bez historii (uzywana np. w eksperymentach RMSE)."""
    steps = xe_idx.shape[0]
    n = xe_idx.shape[1]
    reconstruction = np.zeros((h, w), dtype=np.float32)
    hit_count = np.zeros((h, w), dtype=np.float32)

    for s in range(steps):
        for i in range(n):
            x0 = xe_idx[s, i]
            y0 = ye_idx[s, i]
            backproject_line(reconstruction, hit_count, x0, y0, xd_idx[s, i], yd_idx[s, i], sinogram[s, i])

    for y in range(h):
        for x in range(w):
            if hit_count[y, x] > 0.0:
                reconstruction[y, x] /= hit_count[y, x]

    return reconstruction


@njit(cache=True)
def iradon_transform_with_history(sinogram, xe_idx, ye_idx, xd_idx, yd_idx, h, w):
    """Rekonstrukcja z historia i mapa pokrycia pikseli do normalizacji podgladu."""
    steps = xe_idx.shape[0]
    n = xe_idx.shape[1]
    reconstruction = np.zeros((h, w), dtype=np.float32)
    history = np.zeros((steps, h, w), dtype=np.float32)
    hit_count = np.zeros((h, w), dtype=np.float32)

    for s in range(steps):
        for i in range(n):
            x0 = xe_idx[s, i]
            y0 = ye_idx[s, i]
            backproject_line(reconstruction, hit_count, x0, y0, xd_idx[s, i], yd_idx[s, i], sinogram[s, i])
        history[s, :, :] = reconstruction

    for y in range(h):
        for x in range(w):
            if hit_count[y, x] > 0.0:
                reconstruction[y, x] /= hit_count[y, x]

    return reconstruction, history, hit_count
