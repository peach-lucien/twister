import os
import pickle
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from twister.data.patients import PatientCollection, Patient
from twister.videos.utils import extract_video_details


# ------------------------------- CSV utils -----------------------------------

def save_csv(df: pd.DataFrame, filename: str, folder: str | os.PathLike = "./datasets") -> str:
    """Save DataFrame to CSV under *folder* and return full path."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    # BUGFIX: previously used string concat; now join paths properly
    out = folder_path / filename
    df.to_csv(out)
    return str(out)


def load_from_csv(folder: str | os.PathLike = "./csv_predictions/") -> PatientCollection:
    """Reconstruct a PatientCollection from a folder of CSV prediction files."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"CSV folder not found: {folder_path}")

    csvs = [p.name for p in folder_path.glob("*.csv")]
    patient_ids = list(set([re.split("_mediapipe|_movement", u)[0] for u in csvs]))

    # empty patient list
    patients: List[Patient] = []
    for patient_id in patient_ids:
        p = Patient(patient_id=patient_id)
        p.twister_predictions = {"movement": [], "mediapipe": []}
        patients.append(p)

    pc = PatientCollection()
    pc.add_patient_list(patients)

    # load csvs into patients
    for csv_name in csvs:
        patient_id = re.split("_mediapipe|_movement", csv_name)[0]
        model_out = re.split(patient_id + "_", csv_name)[1].split("_")[0]
        model_type = re.split(patient_id + "_", csv_name)[1].split("_")[1]

        p = pc.get_patient(patient_id)
        df = pd.read_csv(folder_path / csv_name, index_col=0)

        if not p.twister_predictions[model_out]:
            p.twister_predictions[model_out].append({model_type: df})
        else:
            p.twister_predictions[model_out][0][model_type] = df

    return pc


# ----------------------------- object persistence ----------------------------

def save_dataset(obj, filename: str, folder: str | os.PathLike = "./datasets") -> str:
    """Pickle *obj* to <folder>/<filename>.pkl. Returns the file path."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    out = folder_path / f"{filename}.pkl"
    with open(out, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(out)


def load_dataset(filename: str | os.PathLike):
    """Load a dataset from a pickle. Accepts either '.../name.pkl' or any full path."""
    path = Path(filename)
    if path.suffix != ".pkl":
        # allow callers who pass a stem (mirror save_dataset)
        path = path.with_suffix(".pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------ patient builders -----------------------------

def construct_patient_collection(videos: Iterable, patient_ids: Optional[Iterable[str]] = None) -> PatientCollection:
    """Construct a PatientCollection from file paths (or lists of paths)."""
    patients: List[Patient] = []
    vids = list(videos)
    ids = list(patient_ids) if patient_ids is not None else None

    for i, video in enumerate(vids):
        pid = None if ids is None else ids[i]
        patient = construct_patient(video, patient_id=pid)
        patients.append(patient)

    pc = PatientCollection()
    pc.add_patient_list(patients)
    return pc


def construct_patient(video, patient_id: Optional[str] = None) -> Patient:
    """Construct a single Patient object from one path (or a list of paths)."""
    # infer patient_id from filename if not given
    if not patient_id:
        if isinstance(video, list):
            patient_id = Path(video[0]).name.split("_preprocessed")[0]
        else:
            patient_id = Path(video).name.split("_preprocessed")[0]

    # extract video details
    if isinstance(video, list):
        video_details = [extract_video_details(v) for v in video]
    else:
        video_details = [extract_video_details(video)]

    p = Patient(
        sampling_frequency=video_details[0]["fps"],
        patient_id=patient_id,
        video_details=video_details,
    )
    p.twister_predictions = {"movement": None, "mediapipe": None}
    return p


def find_video_files(directory: str | os.PathLike, extensions=None):
    """Return list of video files in *directory* with given extensions."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".wmv", ".mkv", ".MOV"]
    directory = Path(directory)
    files = []
    for item in directory.iterdir():
        if item.is_file() and any(item.name.endswith(ext) for ext in extensions):
            files.append(str(item))
    return files
