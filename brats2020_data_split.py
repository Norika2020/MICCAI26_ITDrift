"""
Create the BraTS 2020 train/validation/test split used in this work.

What this file does:
- load the BraTS name mapping and survival metadata
- merge them on Brats20ID
- create a stratified 70/15/15 split using Grade
- save plain text ID lists
- save an nnU-Net ``splits_final.pkl`` file for the train/val split
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RANDOM_STATE = 42


def load_brats_metadata(
    name_mapping_csv: str | Path,
    survival_csv: str | Path,
    id_column_in_name_map: str = "BraTS_2020_subject_ID",
    merged_id_column: str = "Brats20ID",
) -> pd.DataFrame:
    """
    Load BraTS metadata and merge the two csv files on the case ID.

    Parameters
    ----------
    name_mapping_csv:
        Path to ``name_mapping.csv``.
    survival_csv:
        Path to ``survival_info.csv``.
    id_column_in_name_map:
        Column name used in the name-mapping file for the BraTS case ID.
    merged_id_column:
        Standardised ID column name used after loading.

    Returns
    -------
    pd.DataFrame
        Merged metadata table.
    """
    name_map = pd.read_csv(name_mapping_csv).copy()
    survival = pd.read_csv(survival_csv).copy()

    if id_column_in_name_map in name_map.columns and merged_id_column != id_column_in_name_map:
        name_map = name_map.rename(columns={id_column_in_name_map: merged_id_column})

    if merged_id_column not in name_map.columns:
        raise ValueError(
            f"Could not find ID column '{merged_id_column}' after loading {name_mapping_csv}."
        )
    if merged_id_column not in survival.columns:
        raise ValueError(
            f"Could not find ID column '{merged_id_column}' in {survival_csv}."
        )

    meta = pd.merge(name_map, survival, on=merged_id_column, how="left")

    if "Grade" not in meta.columns:
        raise ValueError("Merged metadata must contain a 'Grade' column for stratified splitting.")

    return meta


def create_stratified_split(
    meta: pd.DataFrame,
    grade_column: str = "Grade",
    id_column: str = "Brats20ID",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """
    Create a stratified train/val/test split.

    The split follows the same two-stage setup used in the notebook:
    first split off the test set, then split the remaining data into
    train and validation.

    Returns
    -------
    dict
        Keys are ``train``, ``val``, and ``test``.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    required = {grade_column, id_column}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    meta = meta.copy()

    train_val_df, test_df = train_test_split(
        meta,
        test_size=test_ratio,
        random_state=random_state,
        stratify=meta[grade_column],
    )

    val_ratio_within_trainval = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_within_trainval,
        random_state=random_state,
        stratify=train_val_df[grade_column],
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def extract_split_ids(
    split_frames: Dict[str, pd.DataFrame],
    id_column: str = "Brats20ID",
) -> Dict[str, List[str]]:
    """
    Pull out the case IDs for each split.
    """
    split_ids: Dict[str, List[str]] = {}
    for split_name, frame in split_frames.items():
        if id_column not in frame.columns:
            raise ValueError(f"Split '{split_name}' is missing ID column '{id_column}'.")
        split_ids[split_name] = frame[id_column].astype(str).tolist()
    return split_ids


def save_id_lists(
    split_ids: Dict[str, List[str]],
    output_dir: str | Path,
    file_template: str = "{split}_ids.txt",
) -> Dict[str, Path]:
    """
    Save each split as a plain text file, one case ID per line.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    for split_name, case_ids in split_ids.items():
        out_path = output_dir / file_template.format(split=split_name)
        out_path.write_text("\n".join(case_ids), encoding="utf-8")
        saved_paths[split_name] = out_path

    return saved_paths


def build_nnunet_split(
    train_ids: List[str],
    val_ids: List[str],
) -> List[Dict[str, List[str]]]:
    """
    Build the single-fold split structure expected by nnU-Net.
    """
    return [{"train": list(train_ids), "val": list(val_ids)}]


def save_nnunet_split(
    train_ids: List[str],
    val_ids: List[str],
    dataset_folder: str | Path,
    filename: str = "splits_final.pkl",
) -> Path:
    """
    Save the nnU-Net train/val split file inside the dataset folder.
    """
    dataset_folder = Path(dataset_folder)
    dataset_folder.mkdir(parents=True, exist_ok=True)

    splits = build_nnunet_split(train_ids=train_ids, val_ids=val_ids)
    out_path = dataset_folder / filename

    with open(out_path, "wb") as handle:
        pickle.dump(splits, handle)

    return out_path


def summarise_split(
    split_frames: Dict[str, pd.DataFrame],
    grade_column: str = "Grade",
) -> pd.DataFrame:
    """
    Small summary table showing split sizes and grade counts.
    """
    rows = []
    for split_name, frame in split_frames.items():
        row = {
            "split": split_name,
            "n_cases": len(frame),
        }
        if grade_column in frame.columns:
            counts = frame[grade_column].value_counts().to_dict()
            for grade, count in counts.items():
                row[f"Grade_{grade}"] = int(count)
        rows.append(row)

    summary = pd.DataFrame(rows).fillna(0)
    count_cols = [col for col in summary.columns if col.startswith("Grade_")]
    for col in count_cols:
        summary[col] = summary[col].astype(int)
    return summary


def create_and_save_brats_split(
    name_mapping_csv: str | Path,
    survival_csv: str | Path,
    output_dir: str | Path,
    nnunet_dataset_folder: str | Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]], pd.DataFrame]:
    """
    Run the full split pipeline and save the outputs.

    Parameters
    ----------
    name_mapping_csv:
        Path to the BraTS name-mapping csv.
    survival_csv:
        Path to the BraTS survival metadata csv.
    output_dir:
        Folder where ``train_ids.txt``, ``val_ids.txt``, and ``test_ids.txt`` will be saved.
    nnunet_dataset_folder:
        Folder where ``splits_final.pkl`` should be saved. If left as None,
        only the plain text ID files are written.
    random_state:
        Random seed for reproducible splitting.

    Returns
    -------
    split_frames, split_ids, summary
    """
    meta = load_brats_metadata(name_mapping_csv=name_mapping_csv, survival_csv=survival_csv)

    split_frames = create_stratified_split(meta=meta, random_state=random_state)
    split_ids = extract_split_ids(split_frames=split_frames)

    save_id_lists(split_ids=split_ids, output_dir=output_dir)

    if nnunet_dataset_folder is not None:
        save_nnunet_split(
            train_ids=split_ids["train"],
            val_ids=split_ids["val"],
            dataset_folder=nnunet_dataset_folder,
        )

    summary = summarise_split(split_frames)
    return split_frames, split_ids, summary


if __name__ == "__main__":
    # Example usage. Change these paths to match your local setup.
    name_mapping_csv = "name_mapping.csv"
    survival_csv = "survival_info.csv"
    output_dir = "."

    # For nnU-Net this should point at the dataset folder itself, e.g.
    # nnUNet_raw/Dataset001_BraTS2020
    nnunet_dataset_folder = None

    _, _, summary = create_and_save_brats_split(
        name_mapping_csv=name_mapping_csv,
        survival_csv=survival_csv,
        output_dir=output_dir,
        nnunet_dataset_folder=nnunet_dataset_folder,
        random_state=DEFAULT_RANDOM_STATE,
    )

    print(summary.to_string(index=False))
