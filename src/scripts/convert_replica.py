"""Build upon: https://github.com/dcharatan/real_estate_10k_tools
https://github.com/donydchen/matchnerf/blob/main/datasets/dtu.py"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--stage", type=str, help="the dataset class, [train, test]")
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def build_camera_info(root_dir):
    """Generate the camera of each png"""
    camera_info_dict = {}
    scene_list = os.listdir(root_dir)
    fx, cx, fy, cy = 320.0, 319.5, 320.0, 239.5
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    near_fars = [0.5, 15.0]
    for scene_i in scene_list:
        for seq_i in os.listdir(os.path.join(root_dir, scene_i)):
            extrinsics_path = os.path.join(root_dir, scene_i, seq_i, "traj_w_c.txt")
            with open(extrinsics_path, "r") as f:
                lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w = torch.from_numpy(c2w).float()
                camera_info_dict[f"{scene_i}-{seq_i}-rgb_{i}"] = {
                    "intrinsics": intrinsic,
                    "extrinsics": c2w,
                    "near_far": near_fars,
                }
    return camera_info_dict


def read_cam_file(filename):
    scale_factor = 1.0 / 200

    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ")
    extrinsic = extrinsic.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ")
    intrinsic = intrinsic.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    near_far = [depth_min, depth_max]
    return intrinsic, extrinsic, near_far


def get_example_keys(root_dir) -> list[str]:
    """Extracted from: https://github.com/donydchen/matchnerf/blob/main/configs/dtu_meta/val_all.txt"""
    scene_list = os.listdir(root_dir)
    return_keys = []
    for scene_i in scene_list:
        for seq_i in os.listdir(os.path.join(root_dir, scene_i)):
            return_keys.append(f"{scene_i}-{seq_i}")
    print(f"Found {len(return_keys)} keys.")
    return return_keys


def get_size(path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path):
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    pic_num = len(os.listdir(example_path))
    all_png_list = [f"rgb_{i}.png" for i in range(pic_num)]
    for png_name in all_png_list:
        img_bin = load_raw(os.path.join(example_path, png_name))
        scene_prefix = example_path.split("/")
        images_dict[f'{scene_prefix[-3]}-{scene_prefix[-2]}-{png_name.split(".")[0]}'] = img_bin
    return images_dict


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(key, camera_info_dict, num=900) -> Metadata:
    timestamps = []
    cameras = []
    url = ""
    for vid in range(num):
        timestamps.append(int(vid))
        intr = camera_info_dict[f"{key}-rgb_{vid}"]["intrinsics"]
        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = 0.5
        saved_cy = 0.5
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = camera_info_dict[f"{key}-rgb_{vid}"]["extrinsics"].inverse()
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":
    # we only use replica for testing, not for training
    stage = args.stage

    camera_info_dict = build_camera_info(INPUT_IMAGE_DIR)

    keys = get_example_keys(INPUT_IMAGE_DIR)

    chunk_size = 0
    chunk_index = 0
    chunk: list[Example] = []

    def save_chunk():
        global chunk_size
        global chunk_index
        global chunk

        chunk_key = f"{chunk_index:0>6}"
        print(f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB).")
        dir = OUTPUT_DIR / stage
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(chunk, dir / f"{chunk_key}.torch")

        # Reset the chunk.
        chunk_size = 0
        chunk_index += 1
        chunk = []

    for key in tqdm(keys):
        key_split = key.split("-")
        image_dir = os.path.join(INPUT_IMAGE_DIR, *key.split("-"), "rgb")
        num_bytes = get_size(image_dir) // 7

        # Read images and metadata ONLY load 50 pictures
        example = load_metadata(key, camera_info_dict)
        images = load_images(image_dir)

        # Merge the images into the example.
        example["images"] = [images[f"{key}-rgb_{timestamp.item()}"] for timestamp in example["timestamps"]]
        assert len(images) == len(example["timestamps"])

        # Add the key to the example.
        example["key"] = key

        print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
        chunk.append(example)
        chunk_size += num_bytes

        if chunk_size >= TARGET_BYTES_PER_CHUNK:
            save_chunk()

    if chunk_size > 0:
        save_chunk()

    # generate index
    print("Generate key:torch index...")
    index = {}
    stage_path = OUTPUT_DIR / stage
    for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
        if chunk_path.suffix == ".torch":
            chunk = torch.load(chunk_path)
            for example in chunk:
                index[example["key"]] = str(chunk_path.relative_to(stage_path))
    with (stage_path / "index.json").open("w") as f:
        json.dump(index, f)
