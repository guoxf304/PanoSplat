"""SynPano panoramic dataset loader.

Each scene directory layout::

    <SynPano_DIR>/<scene_name>/
        images/          # RGB equirectangular images
        depth/           # optional when require_depth=True
        groundtruth.txt  # optional when require_pose=True

Images-only GS training (frozen geometry, merged render)::

    require_depth=False, require_pose=False
    → only ``images/`` is required; placeholder depth, identity poses, all-valid masks.

Full layout uses groundtruth poses (c2w: x,y,z + quaternion) converted to w2c for the batch.
"""

from __future__ import annotations

import logging
import math
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch

from panovggt.Projection import EquirecRotate
from training.data.base_dataset import BaseDataset
from training.data.cache_utils import load_or_build_json_cache
from training.data.dataset_util import erp_target_resolution, threshold_depth_map

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Hamilton quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _c2w_to_w2c(pose_c2w: np.ndarray) -> np.ndarray:
    R_c2w = pose_c2w[:3, :3]
    t_c2w = pose_c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    pose_w2c = np.zeros((3, 4), dtype=np.float32)
    pose_w2c[:3, :3] = R_w2c
    pose_w2c[:3, 3] = t_w2c
    return pose_w2c


class SynPanoDataset(BaseDataset):
    """Loader for SynPano synthetic panoramic scenes."""

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SynPano_DIR: str = "/mnt/gxf/PanoSplat/datasets/SynPano",
        groundtruth_name: str = "groundtruth.txt",
        images_subdir: str = "images",
        depth_subdir: str = "depth",
        min_num_images: int = 2,
        len_train: int = 1000,
        len_test: int = 100,
        expand_ratio: int = 3,
        augmentation: dict | None = None,
        get_nearby: bool | None = None,
        scene_names: list | None = None,
        depth_max: float = 50.0,
        target_resolution: tuple | None = None,
        require_depth: bool = True,
        require_pose: bool = True,
        placeholder_depth: float = 1.0,
    ):
        super().__init__(common_conf=common_conf)

        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        self.training = common_conf.training
        self.get_nearby = get_nearby if get_nearby is not None else common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.expand_ratio = expand_ratio
        self.SynPano_DIR = osp.abspath(SynPano_DIR)
        self.groundtruth_name = groundtruth_name
        self.images_subdir = images_subdir
        self.depth_subdir = depth_subdir
        self.min_num_images = min_num_images
        self.split = split
        self.scene_names = scene_names
        self.depth_max = float(depth_max)
        self.target_resolution = target_resolution
        self.require_depth = bool(require_depth)
        self.require_pose = bool(require_pose)
        self.placeholder_depth = float(placeholder_depth)

        if split == "train":
            self.mode = "train"
            self.dataset_length = len_train
        elif split in ("val", "test", "test_final"):
            self.mode = "val" if split == "val" else "test"
            self.dataset_length = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.augmentation = augmentation if augmentation is not None else common_conf.augs
        self._equi_cache = {}

        t0 = time.time()
        self._load_index_cache()
        logging.info(f"SynPano index ready in {time.time() - t0:.1f}s")

        self.sequence_list_len = len(self.trajectories)
        if self.trajectories:
            self.base_resolution = tuple(self.trajectories[0]["resolution"])
        else:
            self.base_resolution = (518, 1036)
            logging.warning("No SynPano trajectories found.")

        status = "Training" if self.training else "Testing"
        data_mode = self._index_mode_tag()
        logging.info(
            f"{status}: SynPano scenes={self.sequence_list_len}, len={len(self)}, mode={data_mode}"
        )

    def __len__(self):
        return self.dataset_length

    def _load_index_cache(self):
        cache_dir = osp.join(self.SynPano_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        scene_tag = "all" if not self.scene_names else "_".join(sorted(self.scene_names))
        mode_tag = self._index_mode_tag()
        cache_path = osp.join(
            cache_dir, f"SynPano_{self.mode}_{scene_tag}_{mode_tag}_index.json"
        )

        def build_fn():
            return self._build_index()

        self.trajectories = load_or_build_json_cache(cache_path, build_fn)
        self.trajectories = self._validate_trajectories_on_disk(self.trajectories)

    def _rgb_path_for_frame(self, scene_dir: str, frame: dict) -> str:
        return osp.join(scene_dir, self.images_subdir, frame["image_name"])

    def _existing_frame_indices(self, traj: dict) -> list[int]:
        scene_dir = traj["scene_dir"]
        indices = []
        for i, frame in enumerate(traj["frames"]):
            if osp.isfile(self._rgb_path_for_frame(scene_dir, frame)):
                if not self.require_depth or self._resolve_depth_path(
                    scene_dir, frame["stem"]
                ) is not None:
                    indices.append(i)
        return indices

    def _validate_trajectories_on_disk(self, trajs: list) -> list:
        """Drop missing RGB/depth entries from cached trajectories (stale index fix)."""
        refreshed = []
        for traj in trajs:
            scene = traj["scene"]
            scene_dir = traj["scene_dir"]
            rgb_dir = osp.join(scene_dir, self.images_subdir)
            if not osp.isdir(rgb_dir):
                logging.warning("SynPano %s: missing images dir, skip", scene)
                continue

            kept_frames = []
            for frame in traj["frames"]:
                rgb_path = osp.join(rgb_dir, frame["image_name"])
                if not osp.isfile(rgb_path):
                    continue
                if self.require_depth and self._resolve_depth_path(
                    scene_dir, frame["stem"]
                ) is None:
                    continue
                kept_frames.append(frame)

            dropped = len(traj["frames"]) - len(kept_frames)
            if dropped > 0:
                logging.info(
                    "SynPano %s: refreshed index %s -> %s frame(s) on disk",
                    scene,
                    len(traj["frames"]),
                    len(kept_frames),
                )

            if len(kept_frames) < self.min_num_images:
                logging.warning(
                    "SynPano %s: only %s readable frame(s) (< %s), skip",
                    scene,
                    len(kept_frames),
                    self.min_num_images,
                )
                continue

            traj_out = dict(traj)
            traj_out["frames"] = kept_frames
            refreshed.append(traj_out)

        if not refreshed:
            raise RuntimeError(
                "No valid SynPano trajectories after refreshing cached indices. "
                "Check datasets/SynPano/<scene>/images/ or delete stale cache under "
                f"{osp.join(self.SynPano_DIR, 'cache')}."
            )
        return refreshed

    def _build_index(self) -> list:
        if not osp.isdir(self.SynPano_DIR):
            raise FileNotFoundError(f"SynPano_DIR not found: {self.SynPano_DIR}")

        scene_dirs = []
        if self.scene_names:
            for name in self.scene_names:
                path = osp.join(self.SynPano_DIR, name)
                if osp.isdir(path):
                    scene_dirs.append(path)
                else:
                    logging.warning(f"Scene not found, skipping: {path}")
        else:
            for entry in sorted(os.listdir(self.SynPano_DIR)):
                path = osp.join(self.SynPano_DIR, entry)
                if not osp.isdir(path) or entry == "cache":
                    continue
                if self._is_valid_scene_dir(path):
                    scene_dirs.append(path)

        trajs = []
        for scene_dir in scene_dirs:
            scene_name = osp.basename(scene_dir)
            frames = self._load_frames_for_scene(scene_dir)
            if len(frames) < self.min_num_images:
                logging.warning(
                    f"Scene {scene_name} has {len(frames)} frames (< {self.min_num_images}), skip"
                )
                continue

            rgb_dir = osp.join(scene_dir, self.images_subdir)
            if not osp.isdir(rgb_dir):
                logging.warning(f"Missing images dir for {scene_name}: {rgb_dir}")
                continue

            first_rgb = osp.join(rgb_dir, frames[0]["image_name"])
            if not osp.isfile(first_rgb):
                logging.warning(f"Missing first RGB for {scene_name}: {first_rgb}")
                continue

            img_bgr = cv2.imread(first_rgb, cv2.IMREAD_COLOR)
            if img_bgr is None:
                logging.warning(f"Cannot read RGB for {scene_name}: {first_rgb}")
                continue
            h, w = img_bgr.shape[:2]

            trajs.append(
                {
                    "scene": scene_name,
                    "scene_dir": scene_dir,
                    "frames": frames,
                    "resolution": [int(h), int(w)],
                }
            )

        if not trajs:
            raise RuntimeError(
                f"No valid SynPano scenes under {self.SynPano_DIR}. "
                f"Expected layout: <scene>/{self.images_subdir}/"
                + (
                    f" (+ {self.groundtruth_name}, depth/)"
                    if self.require_pose or self.require_depth
                    else ""
                )
            )
        logging.info(f"SynPano indexed {len(trajs)} scene(s)")
        return trajs

    def _index_mode_tag(self) -> str:
        if not self.require_depth and not self.require_pose:
            return "images_only"
        if not self.require_depth:
            return "rgbonly"
        if not self.require_pose:
            return "nopose"
        return "full"

    def _is_valid_scene_dir(self, scene_dir: str) -> bool:
        if not osp.isdir(scene_dir):
            return False
        if self.require_pose:
            if not osp.isfile(osp.join(scene_dir, self.groundtruth_name)):
                return False
        rgb_dir = osp.join(scene_dir, self.images_subdir)
        if not osp.isdir(rgb_dir):
            return False
        return any(
            osp.splitext(name)[1].lower() in _IMAGE_EXTS for name in os.listdir(rgb_dir)
        )

    def _load_frames_for_scene(self, scene_dir: str) -> list:
        if self.require_pose:
            gt_path = osp.join(scene_dir, self.groundtruth_name)
            if not osp.isfile(gt_path):
                logging.warning(f"Missing {self.groundtruth_name} for {scene_dir}")
                return []
            frames = self._parse_groundtruth(gt_path)
            return self._filter_available_frames(scene_dir, frames)
        return self._scan_image_frames(scene_dir)

    def _scan_image_frames(self, scene_dir: str) -> list:
        """Build frame list by scanning ``images/`` (identity placeholder poses)."""
        rgb_dir = osp.join(scene_dir, self.images_subdir)
        image_names = sorted(
            name
            for name in os.listdir(rgb_dir)
            if osp.splitext(name)[1].lower() in _IMAGE_EXTS
        )
        identity_c2w = np.eye(4, dtype=np.float32).tolist()
        frames = []
        for frame_id, image_name in enumerate(image_names):
            stem = osp.splitext(image_name)[0]
            frames.append(
                {
                    "frame_id": frame_id,
                    "image_name": image_name,
                    "stem": stem,
                    "c2w": identity_c2w,
                }
            )
        scene_name = osp.basename(scene_dir)
        logging.info(f"SynPano {scene_name}: scanned {len(frames)} image(s) (no poses)")
        return frames

    def _parse_groundtruth(self, gt_path: str) -> list:
        frames = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    logging.warning(f"Skip malformed pose line: {line}")
                    continue
                frame_id = int(parts[0])
                image_name = parts[1]
                x, y, z = map(float, parts[2:5])
                qx, qy, qz, qw = map(float, parts[5:9])
                R = _quat_xyzw_to_rot(qx, qy, qz, qw)
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = R
                c2w[:3, 3] = [x, y, z]
                stem = osp.splitext(image_name)[0]
                frames.append(
                    {
                        "frame_id": frame_id,
                        "image_name": image_name,
                        "stem": stem,
                        "c2w": c2w.tolist(),
                    }
                )
        return frames

    def _filter_available_frames(self, scene_dir: str, frames: list) -> list:
        """Keep frames whose RGB exists; when require_depth, also require a depth file."""
        rgb_dir = osp.join(scene_dir, self.images_subdir)
        kept = []
        for frame in frames:
            rgb_path = osp.join(rgb_dir, frame["image_name"])
            if not osp.isfile(rgb_path):
                continue
            if self.require_depth and self._resolve_depth_path(scene_dir, frame["stem"]) is None:
                continue
            kept.append(frame)
        dropped = len(frames) - len(kept)
        if dropped > 0:
            scene_name = osp.basename(scene_dir)
            logging.info(
                f"SynPano {scene_name}: indexed {len(kept)} frame(s), "
                f"skipped {dropped} missing rgb/depth"
            )
        return kept

    def _resolve_depth_path(self, scene_dir: str, stem: str) -> str | None:
        depth_dir = osp.join(scene_dir, self.depth_subdir)
        npy_path = osp.join(depth_dir, f"{stem}.npy")
        if osp.isfile(npy_path):
            return npy_path
        png_path = osp.join(depth_dir, f"{stem}.png")
        if osp.isfile(png_path):
            return png_path
        return None

    def _get_equi_rotate(self, equ_h: int):
        rot = self._equi_cache.get(equ_h)
        if rot is None:
            rot = EquirecRotate(equ_h)
            self._equi_cache[equ_h] = rot
        return rot

    def _prepare_augmentation_params(self):
        if not self.training or not self.augmentation:
            return None

        def sample_angle(key, default=60.0):
            if key in self.augmentation and random.random() > 0.5:
                A = float(self.augmentation[key].get("sample_angle", default))
                return (np.random.rand() - 0.5) * 2.0 * A
            return 0.0

        pitch_deg = sample_angle("pitch")
        yaw_deg = sample_angle("yaw")
        roll_deg = sample_angle("roll")
        if abs(pitch_deg) < 1e-6 and abs(yaw_deg) < 1e-6 and abs(roll_deg) < 1e-6:
            return None

        ax = math.radians(pitch_deg)
        ay = math.radians(yaw_deg)
        az = math.radians(roll_deg)

        def Rx(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor(
                [[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32
            )

        def Ry(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor(
                [[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32
            )

        def Rz(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor(
                [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32
            )

        return Rz(az) @ Ry(ay) @ Rx(ax)

    def _read_and_resize_image(self, path, target_resolution):
        h, w = target_resolution
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"cv2.imread failed: {path}")
        if img_bgr.shape[:2] != (h, w):
            img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def _to_single_channel(self, d: np.ndarray) -> np.ndarray:
        if d.ndim == 3:
            if d.shape[2] == 3:
                d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            elif d.shape[2] == 4:
                d = cv2.cvtColor(d, cv2.COLOR_BGRA2GRAY)
            else:
                d = d[..., 0]
        return d

    def _read_and_resize_depth(self, path, target_resolution):
        h, w = target_resolution
        if path.endswith(".npy"):
            d = np.load(path)
            if d.ndim == 3:
                d = d[..., 0]
            d = d.astype(np.float32)
        else:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"cv2.imread failed: {path}")
            d = self._to_single_channel(d).astype(np.float32)
            if d.max() > 20.0:
                d = d / 100.0

        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

        d = threshold_depth_map(
            d, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max
        )
        d[~np.isfinite(d)] = 0.0
        return d[np.newaxis, ...]

    def _placeholder_depth_map(self, target_resolution: tuple) -> np.ndarray:
        h, w = target_resolution
        d = np.full((h, w), self.placeholder_depth, dtype=np.float32)
        return d[np.newaxis, ...]

    def _sample_replace(self, count: int, pool_size: int) -> bool:
        return bool(self.allow_duplicate_img) and count > pool_size

    def _choose_frame_ids(self, valid_indices: list[int], count: int) -> list[int]:
        count = min(int(count), len(valid_indices))
        if count <= 0:
            raise ValueError("count must be positive")
        replace = self._sample_replace(count, len(valid_indices))
        picked = np.random.choice(valid_indices, count, replace=replace)
        return [int(i) for i in np.atleast_1d(picked)]

    def _ensure_unique_frame_ids(
        self,
        ids: list,
        valid_indices: list[int],
        target_count: int,
    ) -> list[int]:
        """Drop duplicates and pad from unused frames when the pool is large enough."""
        target_count = min(int(target_count), len(valid_indices))
        valid_set = set(int(i) for i in valid_indices)
        unique: list[int] = []
        used: set[int] = set()
        for raw in ids:
            idx = int(raw)
            if idx in valid_set and idx not in used:
                unique.append(idx)
                used.add(idx)
        remaining = [i for i in valid_indices if i not in used]
        need = target_count - len(unique)
        if need > 0 and remaining:
            pick = np.random.choice(
                remaining,
                min(need, len(remaining)),
                replace=False,
            )
            unique.extend(int(i) for i in np.atleast_1d(pick))
        return unique

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        aspect_ratio=1.0,
        ids=None,
        seq_name=None,
        geom_aug_R_delta=None,
    ):
        if self.sequence_list_len == 0:
            raise RuntimeError("SynPanoDataset has no trajectories.")

        if seq_index is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        traj = self.trajectories[seq_index % self.sequence_list_len]
        scene = traj["scene"]
        scene_dir = traj["scene_dir"]
        frames = traj["frames"]
        orig_resolution = tuple(traj["resolution"])
        n_frames = len(frames)
        valid_indices = self._existing_frame_indices(traj)
        if len(valid_indices) < 2:
            raise RuntimeError(
                f"SynPano scene {scene} has {len(valid_indices)} readable frame(s); "
                "need at least 2."
            )

        if img_per_seq is None:
            img_per_seq = random.randint(2, min(24, len(valid_indices)))
        img_per_seq = min(img_per_seq, len(valid_indices))

        if ids is None:
            ids = self._choose_frame_ids(valid_indices, img_per_seq)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, n_frames, expand_ratio=self.expand_ratio)
            ids = [int(i) for i in ids if int(i) in valid_indices]
            if len(ids) < 2:
                ids = self._choose_frame_ids(
                    valid_indices,
                    max(2, min(img_per_seq, len(valid_indices))),
                )

        ids = self._ensure_unique_frame_ids(ids, valid_indices, img_per_seq)
        if len(ids) < 2:
            ids = self._choose_frame_ids(
                valid_indices,
                max(2, min(img_per_seq, len(valid_indices))),
            )

        if self.target_resolution is not None:
            target_resolution = tuple(self.target_resolution)
        else:
            target_resolution = erp_target_resolution(self.img_size, self.patch_size)

        equi_rotate = self._get_equi_rotate(target_resolution[0])
        rgb_dir = osp.join(scene_dir, self.images_subdir)
        # One ERP geom-aug rotation per sequence (matches inference_gs rebuild).
        if geom_aug_R_delta is not None:
            if torch.is_tensor(geom_aug_R_delta):
                R_delta = geom_aug_R_delta.detach().cpu().float()
            else:
                R_delta = torch.tensor(geom_aug_R_delta, dtype=torch.float32)
            if (
                R_delta.dim() == 3
                and R_delta.shape[0] == 1
                and R_delta.shape[1:] == (3, 3)
            ):
                R_delta = R_delta[0]
            if R_delta.shape != (3, 3):
                raise ValueError(
                    f"geom_aug_R_delta must be (3, 3), got {tuple(R_delta.shape)}"
                )
            # Match process_one_image: skip rotation when identity.
            eye = torch.eye(3, dtype=R_delta.dtype)
            if torch.allclose(R_delta, eye, atol=1e-5, rtol=0.0):
                R_delta = None
        else:
            R_delta = self._prepare_augmentation_params()

        batch_data = {
            k: []
            for k in [
                "images",
                "depths",
                "extrinsics",
                "cam_points",
                "world_points",
                "point_masks",
                "original_sizes",
                "frame_stems",
            ]
        }
        successful_ids = []

        for idx in ids:
            idx = int(idx)
            frame = frames[idx]
            rgb_path = osp.join(rgb_dir, frame["image_name"])
            if not osp.isfile(rgb_path):
                logging.debug(f"Missing rgb for {scene}/{frame['image_name']}")
                continue
            if self.require_depth:
                depth_path = self._resolve_depth_path(scene_dir, frame["stem"])
                if depth_path is None:
                    logging.debug(
                        f"Missing depth for {scene}/{frame['image_name']}"
                    )
                    continue

            try:
                image = self._read_and_resize_image(rgb_path, target_resolution)
                if self.require_depth:
                    depth_map = self._read_and_resize_depth(
                        depth_path, target_resolution
                    )
                else:
                    depth_map = self._placeholder_depth_map(target_resolution)
                c2w = np.array(frame["c2w"], dtype=np.float32)
                pose_w2c = _c2w_to_w2c(c2w)

                frame_data = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extrinsic_w2c=pose_w2c,
                    shape=target_resolution,
                    equi_rotate=equi_rotate,
                    R_delta=R_delta,
                    depth_max=self.depth_max,
                )
                if not self.require_depth:
                    frame_data["valid_mask"] = torch.ones_like(
                        frame_data["valid_mask"], dtype=torch.bool
                    )

                batch_data["images"].append(frame_data["rgb"])
                batch_data["depths"].append(frame_data["depth_tensor"])
                batch_data["extrinsics"].append(frame_data["extrinsic"])
                batch_data["cam_points"].append(frame_data["cam_coords"])
                batch_data["world_points"].append(frame_data["world_coords"])
                batch_data["point_masks"].append(frame_data["valid_mask"])
                batch_data["original_sizes"].append(np.array(orig_resolution))
                batch_data["frame_stems"].append(frame["stem"])
                successful_ids.append(idx)
            except Exception as e:
                logging.warning(
                    f"Error processing {scene}/{frame['image_name']}: {e}"
                )
                continue

        if len(batch_data["images"]) < 2:
            if len(valid_indices) < 2:
                raise RuntimeError(
                    f"SynPano scene {scene} has fewer than 2 readable frames."
                )
            fallback_ids = self._choose_frame_ids(
                valid_indices,
                min(2, len(valid_indices)),
            )
            logging.warning(
                "SynPano %s: resampling frame ids %s -> %s",
                scene,
                ids,
                fallback_ids,
            )
            return self.get_data(
                seq_index=seq_index,
                img_per_seq=img_per_seq,
                aspect_ratio=aspect_ratio,
                ids=fallback_ids,
            )

        payload = {
            "seq_name": f"SynPano_{scene}",
            "ids": successful_ids,
            "frame_num": len(batch_data["extrinsics"]),
            # Always present so default_collate never KeyErrors across batch items.
            "geom_aug_R_delta": (
                R_delta
                if R_delta is not None
                else torch.eye(3, dtype=torch.float32)
            ),
            **batch_data,
        }
        return payload
