
from __future__ import annotations

import importlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from twister.io import save_csv, save_dataset
from twister.models.cnn_model import load_trained_model, predict_label
from twister.models.mediapipe_landmarks import prepare_empty_dataframe
from procrustes import rotational

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions as mp_solutions

# ─────────────────────────────── data containers ──────────────────────────────

@dataclass(slots=True)
class VideoMeta:
    path: Path
    n_frames: int
    fps: float


@dataclass(slots=True)
class Patient:
    patient_id: str
    video_details: List[VideoMeta]
    twister_predictions: Dict[str, List[pd.DataFrame | dict | None]] = field(
        default_factory=dict
    )


# ────────────────────────────── public entrypoint ─────────────────────────────

def run_all_models(
    tw,
    *,
    save_temp_object: bool = False,
    save_temp_csv: bool = False,
    make_video: bool | Mapping[str, bool] | Iterable[str] = False,
    video_folder: Path | str = "./tracking",
    csv_folder: Path | str = "./csv_predictions",
    recompute_existing: bool = True,
):
    """High‑level orchestration mirroring the original *predict_patients*.

    Accepts both the new :class:`VideoMeta` objects **and** the legacy
    ``dict`` entries (``{"path": ..., "n_frames": ..., "fps": ...}``).
    """

    video_folder = Path(video_folder)
    csv_folder = Path(csv_folder)
    video_folder.mkdir(parents=True, exist_ok=True)
    csv_folder.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------— CNN cache
    cnn_cache: Dict[str, object] = {}
    for name, meta in tw.model_details.items():
        if name == "mediapipe":
            continue
        model_path = (
            importlib.resources.files("twister.models.movement_models")
            / "model_multilabel.pth"
        )
        cnn_cache[name] = load_trained_model(model_path, meta["n_output"])

    # ----------------------------------------------------------------— helpers
    def _checkpoint():
        if save_temp_object:
            save_dataset(tw, "temp", folder="./")

    def _all_exist(paths: Iterable[Path]) -> bool:
        return all(p.exists() for p in paths)

    # ----------------------------------------------------------------— legacy shim
    def _to_meta(v) -> VideoMeta:
        """Convert legacy dicts → :class:`VideoMeta`. Pass through if already ok."""
        if isinstance(v, VideoMeta):
            return v
        if isinstance(v, Mapping):
            return VideoMeta(path=Path(v["path"]), n_frames=int(v["n_frames"]), fps=float(v["fps"]))
        raise TypeError("Unsupported video representation: " + repr(v))

    # ----------------------------------------------------------------— main loop
    for patient in tw.patient_collection:
        patient.twister_predictions = patient.twister_predictions or {}

        for v_idx, raw_video in enumerate(patient.video_details):
            video = _to_meta(raw_video)

            for model_name, meta in tw.model_details.items():

                if patient.twister_predictions.get(model_name) is None:
                    patient.twister_predictions[model_name] = []

                # (a) ─────────────────────────── MEDIAPIPE ────────────────────
                if model_name == "mediapipe":
                    outfile = csv_folder / f"{patient.patient_id}_mediapipe_v{v_idx}.csv"
                    if outfile.exists() and not recompute_existing:
                        patient.twister_predictions[model_name].append(None)
                        continue

                    df = predict_single_video_mediapipe(
                        video,
                        make_video=make_video,
                        video_folder=video_folder,
                    )
                    patient.twister_predictions[model_name].append(df)
                    if save_temp_csv:
                        save_csv(df, outfile.name, folder=csv_folder)

                # (b) ──────────────────────────── CNNs ────────────────────────
                else:
                    stem = f"{patient.patient_id}_{model_name}_v{v_idx}"
                    files = {
                        "preds": csv_folder / f"{stem}_predictions.csv",
                        "probs": csv_folder / f"{stem}_probabilities.csv",
                        "outs":  csv_folder / f"{stem}_outputs.csv",
                    }
                    if _all_exist(files.values()) and not recompute_existing:
                        patient.twister_predictions[model_name].append(None)
                        continue

                    preds, probs, outs = predict_single_video_cnn(video, cnn_cache[model_name])
                    result = {
                        "predictions": pd.DataFrame(preds, columns=meta["label_names"]),
                        "probabilities": pd.DataFrame(probs, columns=meta["label_names"]),
                        "outputs": pd.DataFrame(outs, columns=meta["label_names"]),
                    }
                    patient.twister_predictions[model_name].append(result)

                    if save_temp_csv:
                        for k, df in result.items():
                            save_csv(df, files[k].name, folder=csv_folder)
            _checkpoint()

    save_dataset(tw, "temp", folder="./")
    return tw.patient_collection


# ──────────────────────────────── predictors ─────────────────────────────────

def predict_single_video_cnn(
    video: VideoMeta,
    cnn_model,
):
    """Frame‑wise CNN prediction."""

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise FileNotFoundError(video.path)

    preds, probs, outs = [], [], []
    for _ in tqdm(range(video.n_frames), desc="CNN", leave=False):
        ok, frame = cap.read()
        if not ok:
            break
        out, prb, pred = predict_label(frame, cnn_model)
        preds.append(pred)
        probs.append(prb)
        outs.append(out)
    cap.release()
    return preds, probs, outs


# ════════════════════════════════════════════════════════════════════════════
#                                   MediaPipe
# ════════════════════════════════════════════════════════════════════════════


def predict_single_video_mediapipe(
    video: VideoMeta,
    *,
    make_video: bool | Mapping[str, bool] | Iterable[str] = False,
    video_folder: Path | str = "./tracking",
    plot: bool = False,
) -> pd.DataFrame:
    """Track one video with MediaPipe and return a wide DataFrame."""

    video_on, parts = _parse_video_spec(make_video)
    face_forward = pickle.load(
        importlib.resources.open_binary("twister.models", "average_face_mask.pkl")
    )

    face_mesh, pose, hands = _load_mediapipe_models()

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise FileNotFoundError(video.path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty video")

    # ———————————————————————— writers / placeholders ————————————————————
    writer = None
    if video_on:
        video_folder = Path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w, _ = first.shape
        writer = cv2.VideoWriter(
            str(video_folder / f"{video.path.stem}_MPtracked.mp4"), fourcc, video.fps, (w, h)
        )

    tmpl_df, mapping = prepare_empty_dataframe(hands=True, pose=True, face_mesh=False)
    lm_cols = tmpl_df.columns  # MultiIndex
    lms = pd.DataFrame(index=range(video.n_frames), columns=lm_cols)
    angles = pd.DataFrame(
        index=range(video.n_frames),
        columns=["anteroretrocollis", "torticollis", "laterocollis", "shoulder_angle"],
        dtype=float,
    )
    blends = pd.DataFrame(index=range(video.n_frames))
    
    _FIVE = ("x", "y", "z", "visibility", "presence")
    
    # ——————————————————————————————— main loop ————————————————————————————
    frame = first
    for i in tqdm(range(video.n_frames), desc="MediaPipe", leave=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(i * 1000 / video.fps)
        res_face = face_mesh.detect_for_video(mp_img, ts)
        res_pose = pose.detect_for_video(mp_img, ts)
        res_hands = hands.detect_for_video(mp_img, ts)

        # pose → DataFrame
        if res_pose.pose_world_landmarks:
            for idx, lm in enumerate(res_pose.pose_world_landmarks[0]):
                marker = mapping["pose"][idx]
                for val, sub in zip((lm.x, lm.y, lm.z, lm.visibility, lm.presence), _FIVE):
                    if (marker, sub) in lms.columns:
                        lms.loc[i, (marker, sub)] = val


        # hands → DataFrame
        if res_hands.hand_landmarks:
            for h_idx, hand in enumerate(res_hands.hand_landmarks):
                side = _side_from_handedness(res_hands, h_idx)
                hand_map = mapping[f"{side}_hand"]
                for p_idx, lm in enumerate(hand):
                    marker = hand_map[p_idx]
                    for val, sub in zip((lm.x, lm.y, lm.z, lm.visibility, lm.presence), _FIVE):
                        if (marker, sub) in lms.columns:
                            lms.loc[i, (marker, sub)] = val


        # head & shoulders
        if res_face.face_landmarks:
            face_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in res_face.face_landmarks[0]], dtype=float
            )
            eul, shoulder = _get_head_angle(lms.loc[[i]], face_forward, face_3d)
        else:
            eul, shoulder = [np.nan, np.nan, np.nan], np.nan
        angles.loc[i, ["anteroretrocollis", "torticollis", "laterocollis"]] = eul
        angles.loc[i, "shoulder_angle"] = shoulder

        # blendshapes
        if res_face.face_blendshapes:
            for bs in res_face.face_blendshapes[0]:
                blends.loc[i, bs.category_name] = bs.score

        # drawing
        if video_on or (plot and i % 5 == 0):
            img = frame.copy()
            if parts["face"] and res_face.face_landmarks:
                img = _draw_face(img, res_face)
            if parts["pose"] and res_pose.pose_landmarks:
                img = _draw_pose(img, res_pose)
            if parts["hands"] and res_hands.hand_landmarks:
                img = _draw_hands(img, res_hands)
            if plot and i % 5 == 0:
                import matplotlib.pyplot as plt

                plt.figure(); plt.imshow(img); plt.title(str(eul))
            if video_on:
                writer.write(img)

        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    if writer:
        writer.release()

    return pd.concat([angles, blends, lms], axis=1)


# ─────────────────────────────── helper utils ────────────────────────────────

def _parse_video_spec(spec: bool | Mapping[str, bool] | Iterable[str]):
    default = {"face": True, "pose": True, "hands": True}
    if not spec:
        return False, {}
    if spec is True:
        return True, default
    if isinstance(spec, Mapping):
        return True, {k: bool(spec.get(k, False)) for k in default}
    return True, {k: k in spec for k in default}


def _load_mediapipe_models():
    """Return (face, pose, hands) landmarker objects."""
    Vision = mp.tasks.vision
    rm = Vision.RunningMode.VIDEO

    def _task_path(fname: str) -> Path:
        return Path(importlib.resources.files("twister.models.mediapipe_models")) / fname

    # hands
    hand_opts = Vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_task_path("hand_landmarker.task")),
        num_hands=2,
        running_mode=rm,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = Vision.HandLandmarker.create_from_options(hand_opts)

    # face
    face_opts = Vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_task_path("face_landmarker.task")),
        num_faces=1,
        running_mode=rm,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face = Vision.FaceLandmarker.create_from_options(face_opts)

    # pose
    pose_opts = Vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_task_path("pose_landmarker_heavy.task")),
        num_poses=1,
        running_mode=rm,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = Vision.PoseLandmarker.create_from_options(pose_opts)

    return face, pose, hands


# --- drawing wrappers --------------------------------------------------------


def _draw_face(img, res):
    anno = img.copy()
    for lms in res.face_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in lms])
        mp_solutions.drawing_utils.draw_landmarks(
            anno, proto, mp.solutions.face_mesh.FACEMESH_TESSELATION,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_solutions.drawing_utils.draw_landmarks(
            anno, proto, mp.solutions.face_mesh.FACEMESH_CONTOURS,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_solutions.drawing_utils.draw_landmarks(
            anno, proto, mp.solutions.face_mesh.FACEMESH_IRISES,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return anno


def _draw_pose(img, res):
    """Pretty pose overlay (MediaPipe default colours)."""
    anno = img.copy()
    for lms in res.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in lms
        )

        mp_solutions.drawing_utils.draw_landmarks(
            image=anno,
            landmark_list=proto,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return anno


def _draw_hands(img, res):
    """Pretty hand overlay (MediaPipe default colours)."""
    anno = img.copy()
    for hand_lms in res.hand_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(
            landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in hand_lms
        )

        mp_solutions.drawing_utils.draw_landmarks(
            image=anno,
            landmark_list=proto,
            connections=mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_solutions.drawing_styles.get_default_hand_connections_style(),  # ✔︎ :contentReference[oaicite:1]{index=1}
        )
    return anno


# --- maths -------------------------------------------------------------------

def _get_head_angle(pose_row: pd.DataFrame, face_forward: np.ndarray, face_3d: np.ndarray):
    cols = [
        ("left_shoulder", "x"), ("left_shoulder", "y"),
        ("right_shoulder", "x"), ("right_shoulder", "y"),
    ]
    try:
        if not pose_row[cols].isna().any().any():
            dx = (pose_row[cols[0]] - pose_row[cols[2]]).astype(float).iat[0]
            dy = (pose_row[cols[1]] - pose_row[cols[3]]).astype(float).iat[0]
            ang = -np.arctan2(dy, dx)
            R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            rot_ff = np.hstack([face_forward[:, :2] @ R2, face_forward[:, 2:]])
            shoulder_deg = np.rad2deg(ang)
        else:
            raise ValueError
    except Exception:
        rot_ff = face_forward.copy()
        shoulder_deg = np.nan
    rigid = rotational(rot_ff, face_3d, scale=True, translate=True)
    eul = _rotation_matrix_to_euler(rigid.t)
    return eul, shoulder_deg


def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    if abs(R[2, 0]) != 1.0:
        theta1 = -np.arcsin(R[2, 0])
        theta2 = np.pi - theta1
        psi1 = np.arctan2(R[2, 1] / np.cos(theta1), R[2, 2] / np.cos(theta1))
        psi2 = np.arctan2(R[2, 1] / np.cos(theta2), R[2, 2] / np.cos(theta2))
        phi1 = np.arctan2(R[1, 0] / np.cos(theta1), R[0, 0] / np.cos(theta1))
        phi2 = np.arctan2(R[1, 0] / np.cos(theta2), R[0, 0] / np.cos(theta2))
        sols = np.array([[psi1, theta1, phi1], [psi2, theta2, phi2]]) * 180 / np.pi
        return sols[np.argmin(np.abs(sols.sum(axis=1)))]
    phi = 0.0
    if R[2, 0] == -1.0:
        theta = np.pi / 2
        psi = phi + np.arctan2(R[1, 2], R[1, 1])
    else:
        theta = -np.pi / 2
        psi = -phi + np.arctan2(-R[1, 2], -R[1, 1])
    return np.array([psi, theta, phi]) * 180 / np.pi


# --- misc ---------------------------------------------------------------------

def _side_from_handedness(result, idx: int) -> str:
    if result.handedness and len(result.handedness) > idx:
        return result.handedness[idx][0].category_name.capitalize()
    return "Left" if idx == 0 else "Right"
