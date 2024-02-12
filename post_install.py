
import os
import requests


def download_file(url, target_path):
    # Check if the file already exists
    if not os.path.exists(target_path):
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"File {target_path} already exists. Skipping download.")


def post_install():
    # Define the URLs for the large files and their target locations within the package
    files_to_download = {
        "https://dataverse.harvard.edu/api/access/datafile/8542958": "./twister/models/mediapipe_models/face_landmarker.task",
        "https://dataverse.harvard.edu/api/access/datafile/8542960": "./twister/models/movement_models/model_multilabel.pth",
        "https://dataverse.harvard.edu/api/access/datafile/8542959": "./twister/models/mediapipe_models/pose_landmarker_full.task",
    }

    for url, target_path in files_to_download.items():
        download_file(url, target_path)

if __name__ == "__main__":
    post_install()