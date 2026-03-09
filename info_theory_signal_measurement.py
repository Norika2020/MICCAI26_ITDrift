"""
Utilities for generating synthetic BraTS intensity shifts and measuring
input-level information-theoretic signals.

Main pieces:
- load BraTS cases in nnU-Net format
- generate brightness, Gaussian noise, and bias-field shifts
- save drifted volumes back to nnU-Net style folders
- normalize intensities per modality
- build masked histograms inside the brain region
- compute Jensen-Shannon divergence and entropy-based signals
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi


ArrayLike = np.ndarray
MODALITY_NAMES = ("T1", "T1ce", "T2", "FLAIR")


@dataclass(frozen=True)
class BraTSPaths:
    """
    Paths for BraTS raw data and output folders.
    """

    base_dir: Path
    raw_dataset_dir: Path
    images_dir: Path
    labels_dir: Path

    @classmethod
    def from_base_dir(
        cls,
        base_dir: str | Path,
        dataset_name: str = "Dataset001_BraTS2020",
    ) -> "BraTSPaths":
        base_dir = Path(base_dir)
        raw_dataset_dir = base_dir / "nnUNet_raw" / dataset_name
        return cls(
            base_dir=base_dir,
            raw_dataset_dir=raw_dataset_dir,
            images_dir=raw_dataset_dir / "imagesTr",
            labels_dir=raw_dataset_dir / "labelsTr",
        )


def read_case_ids(case_ids_file: str | Path) -> List[str]:
    """
    Read case IDs from a plain text file, one case per line.
    """
    with open(case_ids_file, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_case(case_id: str, paths: BraTSPaths) -> Tuple[ArrayLike, ArrayLike]:
    """
    Load one 4-modality BraTS case and its label.

    Returns
    -------
    imgs : np.ndarray
        Shape (4, H, W, D), modality order [T1, T1ce, T2, FLAIR].
    label : np.ndarray
        Shape (H, W, D).
    """
    channels: List[ArrayLike] = []
    for channel_idx in range(4):
        img_path = paths.images_dir / f"{case_id}_{channel_idx:04d}.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing modality {channel_idx} for {case_id}: {img_path}")
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        channels.append(img)

    label_path = paths.labels_dir / f"{case_id}.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")
    label = nib.load(str(label_path)).get_fdata().astype(np.int16)

    imgs = np.stack(channels, axis=0)
    return imgs, label


def save_case_modalities(
    case_id: str,
    imgs: ArrayLike,
    output_dir: str | Path,
    paths: BraTSPaths,
) -> None:
    """
    Save a 4-modality case back to nnU-Net naming:
    <case_id>_0000.nii.gz ... <case_id>_0003.nii.gz
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel_idx in range(imgs.shape[0]):
        src_path = paths.images_dir / f"{case_id}_{channel_idx:04d}.nii.gz"
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source modality {channel_idx} for {case_id}: {src_path}")

        src_nii = nib.load(str(src_path))
        out_nii = nib.Nifti1Image(imgs[channel_idx].astype(np.float32), src_nii.affine, src_nii.header)
        out_path = output_dir / f"{case_id}_{channel_idx:04d}.nii.gz"
        nib.save(out_nii, str(out_path))


def robust_minmax_normalize(volume: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    """
    Normalize one volume to [0, 1] using the 1st and 99th percentiles.
    """
    lo = float(np.percentile(volume, 1))
    hi = float(np.percentile(volume, 99))
    volume = np.clip(volume, lo, hi)
    return ((volume - lo) / (hi - lo + eps)).astype(np.float32)


def normalize_modalities(imgs: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    """
    Normalize each modality independently to [0, 1].
    """
    normalized = np.zeros_like(imgs, dtype=np.float32)
    for modality_idx in range(imgs.shape[0]):
        normalized[modality_idx] = robust_minmax_normalize(imgs[modality_idx], eps=eps)
    return normalized


def brain_mask_from_volume(volume: ArrayLike) -> ArrayLike:
    """
    Simple brain mask for BraTS. Background is zero.
    """
    return volume > 0


def brain_mask_from_case(imgs: ArrayLike, modality_idx: int = 3) -> ArrayLike:
    """
    Build a brain mask from one modality. By default this uses FLAIR.
    """
    return brain_mask_from_volume(imgs[modality_idx])


def compute_histogram_masked(
    volume: ArrayLike,
    mask: ArrayLike,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> ArrayLike:
    """
    Compute a normalized histogram from masked voxels only.
    """
    values = volume[mask]
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return np.ones(bins, dtype=np.float64) / bins
    return hist / hist.sum()


def compute_histogram(
    volume: ArrayLike,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> ArrayLike:
    """
    Compute a normalized histogram on the full volume.
    """
    hist, _ = np.histogram(volume.ravel(), bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return np.ones(bins, dtype=np.float64) / bins
    return hist / hist.sum()


def shannon_entropy(probabilities: ArrayLike, eps: float = 1e-12) -> float:
    """
    Shannon entropy of a discrete distribution.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64) + eps
    probabilities = probabilities / probabilities.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def js_divergence(p: ArrayLike, q: ArrayLike, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions.
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def modality_histograms_masked(
    imgs: ArrayLike,
    mask: ArrayLike,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> List[ArrayLike]:
    """
    Compute one masked histogram per modality.
    """
    return [
        compute_histogram_masked(imgs[modality_idx], mask, bins=bins, value_range=value_range)
        for modality_idx in range(imgs.shape[0])
    ]


def modality_js_divergence_masked(
    imgs_ref: ArrayLike,
    imgs_shifted: ArrayLike,
    mask: ArrayLike,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """
    Per-modality JS divergence between a reference case and a shifted case.
    """
    ref_hists = modality_histograms_masked(imgs_ref, mask, bins=bins, value_range=value_range)
    shifted_hists = modality_histograms_masked(imgs_shifted, mask, bins=bins, value_range=value_range)
    return {
        MODALITY_NAMES[idx]: js_divergence(ref_hist, shifted_hist)
        for idx, (ref_hist, shifted_hist) in enumerate(zip(ref_hists, shifted_hists))
    }


def modality_entropy_masked(
    imgs: ArrayLike,
    mask: ArrayLike,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """
    Per-modality entropy from masked intensity histograms.
    """
    hists = modality_histograms_masked(imgs, mask, bins=bins, value_range=value_range)
    return {
        MODALITY_NAMES[idx]: shannon_entropy(hist)
        for idx, hist in enumerate(hists)
    }


def case_level_input_signals(
    imgs_ref: ArrayLike,
    imgs_shifted: ArrayLike,
    mask: Optional[ArrayLike] = None,
    bins: int = 64,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, float]:
    """
    Compute case-level input signals from normalized images.

    Returns average and per-modality values for JS divergence, entropy on the
    shifted case, and entropy change relative to the reference case.
    """
    if mask is None:
        mask = brain_mask_from_case(imgs_ref)

    js_per_modality = modality_js_divergence_masked(
        imgs_ref, imgs_shifted, mask=mask, bins=bins, value_range=value_range
    )
    entropy_ref = modality_entropy_masked(imgs_ref, mask=mask, bins=bins, value_range=value_range)
    entropy_shifted = modality_entropy_masked(imgs_shifted, mask=mask, bins=bins, value_range=value_range)

    signals: Dict[str, float] = {}
    signals["js_mean"] = float(np.mean(list(js_per_modality.values())))
    signals["input_entropy_mean"] = float(np.mean(list(entropy_shifted.values())))
    signals["delta_input_entropy_mean"] = float(
        np.mean([entropy_shifted[name] - entropy_ref[name] for name in MODALITY_NAMES])
    )

    for modality_name in MODALITY_NAMES:
        signals[f"js_{modality_name.lower()}"] = js_per_modality[modality_name]
        signals[f"input_entropy_{modality_name.lower()}"] = entropy_shifted[modality_name]
        signals[f"delta_input_entropy_{modality_name.lower()}"] = (
            entropy_shifted[modality_name] - entropy_ref[modality_name]
        )

    return signals


def make_seed(case_id: str, severity: float, kind: str) -> int:
    """
    Deterministic seed from case ID, severity value, and drift kind.
    """
    key = f"{case_id}_{kind}_{severity}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % (2**32)


def apply_brightness_shift(imgs: ArrayLike, delta: float) -> ArrayLike:
    """
    Add a brightness shift in raw intensity space inside the non-zero region.
    The shift is scaled by the robust intensity range of each modality.
    """
    shifted = np.zeros_like(imgs, dtype=np.float32)
    for modality_idx in range(imgs.shape[0]):
        volume = imgs[modality_idx].astype(np.float32)
        lo, hi = np.percentile(volume, (1, 99))
        shift = delta * (hi - lo)
        out = volume.copy()
        mask = volume > 0
        out[mask] = volume[mask] + shift
        shifted[modality_idx] = out
    return shifted


def apply_gaussian_noise(imgs: ArrayLike, alpha: float, rng: Optional[np.random.RandomState] = None) -> ArrayLike:
    """
    Add zero-mean Gaussian noise in raw intensity space inside the brain region.
    Noise std is alpha times the robust intensity range.
    """
    if rng is None:
        rng = np.random.RandomState()

    shifted = np.zeros_like(imgs, dtype=np.float32)
    for modality_idx in range(imgs.shape[0]):
        volume = imgs[modality_idx].astype(np.float32)
        lo, hi = np.percentile(volume, (1, 99))
        sigma = alpha * (hi - lo)
        noise = rng.normal(0.0, sigma, size=volume.shape)
        out = volume.copy()
        mask = volume > 0
        out[mask] = volume[mask] + noise[mask]
        shifted[modality_idx] = out
    return shifted


def generate_bias_field(
    shape: Sequence[int],
    beta: float,
    rng: Optional[np.random.RandomState] = None,
    sigma_factor: float = 0.15,
) -> ArrayLike:
    """
    Generate a smooth multiplicative bias field for one 3D volume.
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w, d = shape
    field = rng.normal(0.0, 1.0, size=(h, w, d)).astype(np.float32)
    sigma = (h * sigma_factor, w * sigma_factor, d * sigma_factor)
    field = ndi.gaussian_filter(field, sigma=sigma, mode="reflect")
    field -= field.mean()
    field /= field.std() + 1e-8
    return (1.0 + beta * field).astype(np.float32)


def apply_bias_field(imgs: ArrayLike, beta: float, rng: Optional[np.random.RandomState] = None) -> ArrayLike:
    """
    Apply one smooth multiplicative bias field to all modalities inside the brain.
    """
    if rng is None:
        rng = np.random.RandomState()

    _, h, w, d = imgs.shape
    bias_field = generate_bias_field((h, w, d), beta=beta, rng=rng)

    shifted = np.zeros_like(imgs, dtype=np.float32)
    for modality_idx in range(imgs.shape[0]):
        volume = imgs[modality_idx].astype(np.float32)
        out = volume.copy()
        mask = volume > 0
        out[mask] = volume[mask] * bias_field[mask]
        shifted[modality_idx] = out
    return shifted


def generate_brightness_dataset(
    case_ids: Sequence[str],
    paths: BraTSPaths,
    output_root: str | Path,
    deltas: Sequence[float],
) -> None:
    """
    Create brightness-shifted copies of the requested cases.
    """
    output_root = Path(output_root)
    for delta in deltas:
        tag = f"brightness_{int(delta * 100):03d}"
        output_dir = output_root / tag / "imagesTs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for case_id in case_ids:
            imgs, _ = load_case(case_id, paths)
            shifted = apply_brightness_shift(imgs, delta=delta)
            save_case_modalities(case_id, shifted, output_dir, paths)


def generate_noise_dataset(
    case_ids: Sequence[str],
    paths: BraTSPaths,
    output_root: str | Path,
    alphas: Sequence[float],
) -> None:
    """
    Create Gaussian-noise shifted copies of the requested cases.
    """
    output_root = Path(output_root)
    for alpha in alphas:
        tag = f"noise_{int(alpha * 100):03d}"
        output_dir = output_root / tag / "imagesTs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for case_id in case_ids:
            imgs, _ = load_case(case_id, paths)
            seed = make_seed(case_id, severity=alpha, kind="noise")
            rng = np.random.RandomState(seed)
            shifted = apply_gaussian_noise(imgs, alpha=alpha, rng=rng)
            save_case_modalities(case_id, shifted, output_dir, paths)


def generate_bias_dataset(
    case_ids: Sequence[str],
    paths: BraTSPaths,
    output_root: str | Path,
    betas: Sequence[float],
) -> None:
    """
    Create bias-field shifted copies of the requested cases.
    """
    output_root = Path(output_root)
    for beta in betas:
        tag = f"bias_{int(beta * 100):03d}"
        output_dir = output_root / tag / "imagesTs"
        output_dir.mkdir(parents=True, exist_ok=True)

        for case_id in case_ids:
            imgs, _ = load_case(case_id, paths)
            seed = make_seed(case_id, severity=beta, kind="bias")
            rng = np.random.RandomState(seed)
            shifted = apply_bias_field(imgs, beta=beta, rng=rng)
            save_case_modalities(case_id, shifted, output_dir, paths)


def measure_signals_for_shifted_case(
    case_id: str,
    paths: BraTSPaths,
    shift_kind: str,
    severity: float,
    bins: int = 64,
) -> Dict[str, float]:
    """
    Compute input-level information signals for one case after applying a synthetic shift.

    This is useful when we want signal measurement without saving the drifted case to disk.
    """
    imgs_raw, _ = load_case(case_id, paths)
    imgs_ref = normalize_modalities(imgs_raw)

    if shift_kind == "brightness":
        imgs_shifted_raw = apply_brightness_shift(imgs_raw, delta=severity)
    elif shift_kind == "noise":
        rng = np.random.RandomState(make_seed(case_id, severity=severity, kind="noise"))
        imgs_shifted_raw = apply_gaussian_noise(imgs_raw, alpha=severity, rng=rng)
    elif shift_kind == "bias":
        rng = np.random.RandomState(make_seed(case_id, severity=severity, kind="bias"))
        imgs_shifted_raw = apply_bias_field(imgs_raw, beta=severity, rng=rng)
    else:
        raise ValueError(f"Unknown shift_kind: {shift_kind}")

    imgs_shifted = normalize_modalities(imgs_shifted_raw)
    mask = brain_mask_from_case(imgs_ref)
    signals = case_level_input_signals(imgs_ref, imgs_shifted, mask=mask, bins=bins)
    signals["case_id"] = case_id
    signals["shift_kind"] = shift_kind
    signals["severity"] = float(severity)
    return signals


__all__ = [
    "BraTSPaths",
    "MODALITY_NAMES",
    "apply_bias_field",
    "apply_brightness_shift",
    "apply_gaussian_noise",
    "brain_mask_from_case",
    "brain_mask_from_volume",
    "case_level_input_signals",
    "compute_histogram",
    "compute_histogram_masked",
    "generate_bias_dataset",
    "generate_bias_field",
    "generate_brightness_dataset",
    "generate_noise_dataset",
    "js_divergence",
    "load_case",
    "make_seed",
    "measure_signals_for_shifted_case",
    "modality_entropy_masked",
    "modality_histograms_masked",
    "modality_js_divergence_masked",
    "normalize_modalities",
    "read_case_ids",
    "robust_minmax_normalize",
    "save_case_modalities",
    "shannon_entropy",
]
