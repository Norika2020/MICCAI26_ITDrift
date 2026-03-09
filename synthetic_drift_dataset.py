"""
Generate drifted BraTS cases in nnU-Net format.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from info_theory_signal_measurement import (
    BraTSPaths,
    apply_bias_field,
    apply_brightness_shift,
    apply_gaussian_noise,
    load_case,
    read_case_ids,
    save_case_modalities,
)


DRIFT_OUTPUT_FOLDERS: Mapping[str, str] = {
    "brightness": "drifted_images_brightness",
    "noise": "drifted_images_noise",
    "bias": "drifted_images_bias",
}


@dataclass(frozen=True)
class SyntheticDriftConfig:
    """
    Configuration for one drift family.
    """

    drift_type: str
    severities: Sequence[float]
    output_root_name: str | None = None

    def output_root(self) -> str:
        if self.output_root_name is not None:
            return self.output_root_name
        if self.drift_type not in DRIFT_OUTPUT_FOLDERS:
            raise ValueError(
                f"Unknown drift type '{self.drift_type}'. "
                f"Expected one of {sorted(DRIFT_OUTPUT_FOLDERS)}."
            )
        return DRIFT_OUTPUT_FOLDERS[self.drift_type]


def severity_tag(drift_type: str, severity: float) -> str:
    """
    Turn a severity value into the folder tag used in the notebook runs.

    Examples
    --------
    brightness + 0.05 -> brightness_005
    noise + 0.10      -> noise_010
    bias + 0.40       -> bias_040
    """
    return f"{drift_type}_{int(round(float(severity) * 100)):03d}"


def _validate_case_ids(case_ids: Iterable[str]) -> List[str]:
    case_ids = [case_id.strip() for case_id in case_ids if case_id and case_id.strip()]
    if not case_ids:
        raise ValueError("No case IDs were provided.")
    return case_ids


def _apply_drift(
    imgs: np.ndarray,
    case_id: str,
    drift_type: str,
    severity: float,
) -> np.ndarray:
    """
    Dispatch to the right drift function.
    """
    if drift_type == "brightness":
        return apply_brightness_shift(imgs, delta=float(severity))
    if drift_type == "noise":
        return apply_gaussian_noise(imgs, alpha=float(severity))
    if drift_type == "bias":
        return apply_bias_field(imgs, beta=float(severity))
    raise ValueError(
        f"Unknown drift type '{drift_type}'. Expected one of ['brightness', 'noise', 'bias']."
    )


def generate_drift_dataset(
    paths: BraTSPaths,
    case_ids: Sequence[str],
    drift_type: str,
    severities: Sequence[float],
    output_root: str | Path | None = None,
    image_subdir: str = "imagesTs",
) -> Dict[str, Path]:
    """
    Generate one whole drift family and save the cases to disk.

    Parameters
    ----------
    paths:
        BraTS path bundle.
    case_ids:
        Case IDs to process.
    drift_type:
        One of 'brightness', 'noise', or 'bias'.
    severities:
        Drift strengths to apply.
    output_root:
        Root folder for this drift family. If left as None, a default folder
        name is chosen under ``paths.base_dir``.
    image_subdir:
        Usually ``imagesTs`` so the result can be dropped straight into the
        nnU-Net test-time layout.

    Returns
    -------
    dict
        Maps each generated severity tag to its output directory.
    """
    case_ids = _validate_case_ids(case_ids)

    if output_root is None:
        output_root = paths.base_dir / DRIFT_OUTPUT_FOLDERS[drift_type]
    else:
        output_root = Path(output_root)

    output_dirs: Dict[str, Path] = {}

    for severity in severities:
        tag = severity_tag(drift_type, severity)
        output_dir = output_root / tag / image_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[tag] = output_dir

        for case_id in case_ids:
            imgs, _ = load_case(case_id, paths=paths)
            drifted = _apply_drift(imgs, case_id=case_id, drift_type=drift_type, severity=float(severity))
            save_case_modalities(case_id, drifted, output_dir=output_dir, paths=paths)

    return output_dirs


def generate_from_config(
    paths: BraTSPaths,
    case_ids: Sequence[str],
    config: SyntheticDriftConfig,
    image_subdir: str = "imagesTs",
) -> Dict[str, Path]:
    """
    Convenience wrapper for a dataclass config.
    """
    output_root = paths.base_dir / config.output_root()
    return generate_drift_dataset(
        paths=paths,
        case_ids=case_ids,
        drift_type=config.drift_type,
        severities=config.severities,
        output_root=output_root,
        image_subdir=image_subdir,
    )


def generate_all_default_drift_sets(
    paths: BraTSPaths,
    case_ids: Sequence[str],
    image_subdir: str = "imagesTs",
) -> Dict[str, Dict[str, Path]]:
    """
    Regenerate the three drift families used in the experiments.

    The defaults follow the notebook:
    - brightness: [0.05, 0.10, 0.15]
    - noise:      [0.05, 0.10, 0.20]
    - bias:       [0.20, 0.40, 0.60]
    """
    configs = [
        SyntheticDriftConfig("brightness", [0.05, 0.10, 0.15]),
        SyntheticDriftConfig("noise", [0.05, 0.10, 0.20]),
        SyntheticDriftConfig("bias", [0.20, 0.40, 0.60]),
    ]

    generated: Dict[str, Dict[str, Path]] = {}
    for config in configs:
        generated[config.drift_type] = generate_from_config(
            paths=paths,
            case_ids=case_ids,
            config=config,
            image_subdir=image_subdir,
        )
    return generated


def build_paths_from_base_dir(
    base_dir: str | Path,
    dataset_name: str = "Dataset001_BraTS2020",
) -> BraTSPaths:
    """
    Small helper so this file can be used directly without importing the path
    dataclass elsewhere first.
    """
    return BraTSPaths.from_base_dir(base_dir=base_dir, dataset_name=dataset_name)


def generate_from_case_id_file(
    base_dir: str | Path,
    case_ids_file: str | Path,
    drift_type: str,
    severities: Sequence[float],
    dataset_name: str = "Dataset001_BraTS2020",
    image_subdir: str = "imagesTs",
) -> Dict[str, Path]:
    """
    Minimal entry point for scripts.

    Example
    -------
    >>> generate_from_case_id_file(
    ...     base_dir="/path/to/InfoTheo_dataset",
    ...     case_ids_file="/path/to/test_ids.txt",
    ...     drift_type="brightness",
    ...     severities=[0.05, 0.10, 0.15],
    ... )
    """
    paths = build_paths_from_base_dir(base_dir=base_dir, dataset_name=dataset_name)
    case_ids = read_case_ids(case_ids_file)
    return generate_drift_dataset(
        paths=paths,
        case_ids=case_ids,
        drift_type=drift_type,
        severities=severities,
        image_subdir=image_subdir,
    )


def describe_generated_outputs(output_dirs: Mapping[str, Path]) -> str:
    """
    Return a short plain-text summary of the generated folders.
    """
    lines = []
    for tag, output_dir in output_dirs.items():
        lines.append(f"{tag}: {output_dir}")
    return "\n".join(lines)


if __name__ == "__main__":
    BASE_DIR = "/path/to/InfoTheo_dataset"
    TEST_IDS_FILE = "/path/to/test_ids.txt"

    paths = build_paths_from_base_dir(BASE_DIR)
    case_ids = read_case_ids(TEST_IDS_FILE)

    outputs = generate_all_default_drift_sets(paths=paths, case_ids=case_ids)
    for drift_type, folders in outputs.items():
        print(f"\nGenerated {drift_type} folders:")
        print(describe_generated_outputs(folders))
