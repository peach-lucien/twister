from __future__ import annotations

import json
import os
from pathlib import Path
import importlib.resources
import pandas as pd

from twister.videos.preprocess import get_video_filenames, preprocess_videos
from twister.io import construct_patient_collection, load_dataset, find_video_files
from twister.models.predictions import run_all_models
from twister.statistics.statistics import extract
from twister.plotting.plotting import plot


class twstr:
    """Main Twister pipeline object."""

    def __init__(
        self,
        video_path: str | Path,
        *,
        output_path: str | Path = "./outputs",
        patient_collection=None,
        patient_ids=None,
        video_files=None,
        model_details=None,
        model_directory: str | Path | None = None,
        plotting_args: dict | None = None,
    ):
        self.video_path = Path(video_path).resolve()
        self.output_path = Path(output_path).resolve()

        # output subfolders
        self.tracking_dir = self.output_path / "tracking"
        self.csv_dir = self.output_path / "csv"

        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.patient_collection = patient_collection
        self.patient_ids = patient_ids
        self.video_files = video_files

        # model info
        class_dir = Path(__file__).parent
        self.model_directory = (
            Path(model_directory) if model_directory else class_dir / "models" / "mediapipe_models"
        )
        self.model_details = model_details or self._load_model_details()

        # plotting defaults
        self.plotting_args = plotting_args or {
            "plotting_folder": str(self.output_path / "plots"),
            "ext": ".svg",
        }

        # discover videos if not passed
        if self.video_path and not self.video_files:
            self.video_files = find_video_files(self.video_path)

    # ------------------------------------------------------------------
    # Pipeline entry
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        preprocess: bool = False,
        make_video: bool = False,
        save_csv: bool = True,
        save_object: bool = True,
        recompute_existing: bool = True,
    ):
        """Run the full pipeline."""
        if preprocess:
            self.preprocess_videos()

        self.load_data()
        self.predict(
            make_video=make_video,
            save_csv=save_csv,
            save_object=save_object,
            recompute_existing=recompute_existing,
        )
        self.analyse()
        self.aggregate()
        self.plot()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def load_data(self):
        """Build patient collection from discovered video files."""
        self.patient_collection = construct_patient_collection(self.video_files, self.patient_ids)

    def preprocess_videos(self):
        """Resize / clean raw videos before processing."""
        videos = get_video_filenames(self.video_path)
        self.video_files = preprocess_videos(videos)

    def predict(
        self,
        *,
        file=None,
        save_csv: bool = True,
        save_object: bool = True,
        make_video: bool = False,
        recompute_existing: bool = True,
    ):
        """Run DL models and tracking on each patient/video."""
        if file is not None:
            self = load_dataset(file)

        self.patient_collection = run_all_models(
            self,
            save_temp_csv=save_csv,
            save_temp_object=save_object,
            make_video=make_video,
            tracking_folder=self.tracking_dir,
            csv_folder=self.csv_dir,
            recompute_existing=recompute_existing,
        )

    def analyse(self):
        """Extract statistics from patient predictions."""
        self.results = extract(self.patient_collection)

    def aggregate(self):
        """Aggregate feature vectors across patients into a single matrix."""
        all_features = {}
        for patient, result in self.results.items():
            features = [result[feat]["feature_vector"] for feat in result]
            all_features[patient] = pd.concat(features, axis=1)

        feature_matrix = pd.concat(all_features, ignore_index=True)
        feature_matrix.index = list(all_features.keys())
        self.feature_matrix = feature_matrix.astype(float)

    def plot(self):
        """Generate summary plots for each patient."""
        plot(self.results, self.patient_collection, self.plotting_args)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_details(self) -> dict:
        """Read bundled model metadata JSON."""
        with importlib.resources.open_text("twister.models", "model_details.json") as f:
            details = json.load(f)
        for model in details:
            details[model]["model_path"] = str(
                Path(self.model_directory)
                / details[model]["model_directory"]
                / details[model]["model_name"]
            )
        return details
    
    # inside class twstr

    def save(self, name: str = "twister_run") -> str:
        """
        Pickle the whole twstr object under <output_path>/artifacts/<name>.pkl.
        Returns the file path (string).
        """
        artifacts = self.output_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        from twister.io import save_dataset
        save_dataset(self, name, folder=str(artifacts))
        return str(artifacts / f"{name}.pkl")

    @classmethod
    def load(cls, file: str | os.PathLike) -> "twstr":
        """
        Load a previously saved twstr object (path to the .pkl).
        """
        from twister.io import load_dataset
        obj = load_dataset(str(file))
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)!r}, expected {cls.__name__}")
        return obj

