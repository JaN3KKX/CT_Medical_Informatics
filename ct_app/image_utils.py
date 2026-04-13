from pathlib import Path

import numpy as np
from skimage import color, transform


def create_filter(size):
    """Tworzy filtr splotowy typu ramp."""
    kernel = np.zeros(size)
    center = size // 2
    for i in range(size):
        k = i - center
        if k == 0:
            kernel[i] = 1.0
        elif k % 2 != 0:
            kernel[i] = -4.0 / (np.pi**2 * k**2)
    return kernel


def filter_sinogram(sinogram):
    """Naklada splot 1D na kazdy wiersz sinogramu."""
    kernel_size = 21
    kernel = create_filter(kernel_size)
    filtered = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        filtered[i, :] = np.convolve(sinogram[i, :], kernel, mode="same")
    return filtered


def calculate_rmse(img1, img2):
    """Oblicza blad sredniokwadratowy (RMSE)."""
    i1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1) + 1e-8)
    i2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2) + 1e-8)
    return np.sqrt(np.mean((i1 - i2) ** 2))


def resolve_sample_dir():
    """Zwraca katalog z obrazami probkowymi (sprawdza pakiet i folder wyzej)."""
    base_dir = Path(__file__).resolve().parent
    candidate_dirs = [
        base_dir / "tomograf-obrazy",
        base_dir / "tomograf_obrazy",
        base_dir.parent / "tomograf-obrazy",
        base_dir.parent / "tomograf_obrazy",
    ]
    for sample_dir in candidate_dirs:
        if sample_dir.exists() and sample_dir.is_dir():
            return sample_dir
    return candidate_dirs[0]


def list_sample_images(sample_dir):
    """Zwraca posortowana liste obrazow z katalogu probek."""
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not sample_dir.exists():
        return []

    sample_files = []
    for path in sample_dir.iterdir():
        if path.is_file() and path.suffix.lower() in supported_ext:
            sample_files.append(path.name)

    return sorted(sample_files, key=lambda name: name.lower())


def preprocess_image(raw_image):
    """Przeksztalca obraz wejsciowy do skali szarosci float32 i normalizuje."""
    if raw_image.ndim == 2:
        gray = raw_image
    elif raw_image.ndim == 3:
        if raw_image.shape[2] == 4:
            raw_image = color.rgba2rgb(raw_image)
        gray = color.rgb2gray(raw_image)
    else:
        raise ValueError("Nieobslugiwany ksztalt obrazu. Uzyj standardowego obrazu 2D lub RGB.")

    gray = gray.astype(np.float32)
    gray_min = float(np.min(gray))
    gray_max = float(np.max(gray))
    if gray_max > gray_min:
        gray = (gray - gray_min) / (gray_max - gray_min)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    resized = transform.resize(
        gray,
        (128, 128),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    return np.clip(resized, 0.0, 1.0)
