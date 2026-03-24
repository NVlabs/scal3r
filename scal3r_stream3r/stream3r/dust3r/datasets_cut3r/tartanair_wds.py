# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os.path as osp
import numpy as np
import cv2
import numpy as np
import itertools
import os
import sys
import json
import io
import zlib
import tarfile
import struct
from collections import OrderedDict

from stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from stream3r.dust3r.utils.image import imread_cv2


def _build_offset_index(tar_path):
    """Scan a tar file once, return {member_name: (data_offset, data_size)}."""
    index = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.isfile():
                index[member.name] = (member.offset_data, member.size)
    return index


def _load_or_build_shard_index(shard_path):
    """Load cached offset index, or build and save it on first access."""
    cache_path = shard_path + ".offsets"
    if osp.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    # Build by scanning the tar once
    index = _build_offset_index(shard_path)
    # Save for next time (best-effort, ignore write errors)
    try:
        with open(cache_path, "w") as f:
            json.dump(index, f)
    except OSError:
        pass
    return index


class TartanAirWds_Multi(BaseMultiViewDataset):
    """TartanAir dataset backed by WebDataset tar shards.

    Drop-in replacement for TartanAir_Multi. Reads from tar shards + index.json
    instead of individual files for better I/O on distributed filesystems.

    Uses pre-built byte-offset indices for O(1) random access into tar shards,
    avoiding the cost of tarfile.open() scanning on every worker fork.

    Expects the output of datasets_preprocess/convert_tartanair_to_wds.py:
        ROOT/
            index.json
            tartanair-000000.tar
            ...
    """

    def __init__(self, ROOT, *args, max_shard_size=1000, max_open_shards=None, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.max_interval = 20
        self.max_shard_size = max_shard_size
        super().__init__(*args, **kwargs)
        # loading all
        assert self.split is None
        self._load_data()
        # Offset indices: shard_idx -> {member_name: (data_offset, data_size)}
        # Pre-built for ALL shards in main process; workers inherit via fork.
        self._shard_offsets = {}
        self._preload_all_offsets()
        # Auto-size LRU cache to cover all shards (no eviction = no re-open on Lustre)
        num_shards = len(self._shard_offsets)
        self._max_open_shards = max_open_shards if max_open_shards is not None else max(num_shards, 32)
        # Per-worker file handle LRU cache (reset on fork)
        self._fh_cache = OrderedDict()

    def _load_data(self):
        index_path = osp.join(self.ROOT, "index.json")
        with open(index_path) as f:
            index = json.load(f)

        offset = 0
        global_idx = 0  # position in the shard stream (incl. skipped seqs)
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        local_to_shard = []
        local_to_key = []
        j = 0

        for seq in index["sequences"]:
            num_imgs = seq["num_frames"]
            basenames = seq["basenames"]
            keys = seq["keys"]
            cut_off = (
                self.num_views
                if not self.allow_repeat
                else max(self.num_views // 3, 3)
            )

            if num_imgs < cut_off:
                print(f"Skipping {seq['scene']}")
                global_idx += num_imgs
                continue
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scene_dir = seq["seq_dir"]
            scenes.append(scene_dir)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)

            for k in range(num_imgs):
                shard_idx = (global_idx + k) // self.max_shard_size
                local_to_shard.append(shard_idx)
                local_to_key.append(keys[k])

            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            global_idx += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self._local_to_shard = local_to_shard
        self._local_to_key = local_to_key

    def _preload_all_offsets(self):
        """Pre-load offset indices for ALL shards in main process.
        Workers inherit the populated dict via fork, avoiding per-worker
        Lustre metadata reads that cause epoch-boundary spikes."""
        unique_shards = sorted(set(self._local_to_shard))
        print(f"[TartanAirWds] Pre-loading offset indices for {len(unique_shards)} shards ...")
        for shard_idx in unique_shards:
            shard_path = osp.join(self.ROOT, f"tartanair-{shard_idx:06d}.tar")
            self._shard_offsets[shard_idx] = _load_or_build_shard_index(shard_path)
        print(f"[TartanAirWds] Done. {len(self._shard_offsets)} shard offsets loaded.")

    # ------------------------------------------------------------------
    # Byte-offset I/O helpers
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Don't pickle open file handles (they break across fork).
        Offset indices are plain dicts and survive fork safely."""
        state = self.__dict__.copy()
        state["_fh_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _get_shard_offsets(self, shard_idx):
        """Return offset index for a shard, building/loading on first access."""
        if shard_idx not in self._shard_offsets:
            shard_path = osp.join(self.ROOT, f"tartanair-{shard_idx:06d}.tar")
            self._shard_offsets[shard_idx] = _load_or_build_shard_index(shard_path)
        return self._shard_offsets[shard_idx]

    def _get_fh(self, shard_idx):
        """Return an open file handle for a shard, using an LRU cache."""
        if shard_idx in self._fh_cache:
            self._fh_cache.move_to_end(shard_idx)
            return self._fh_cache[shard_idx]

        # Evict oldest if cache is full
        if len(self._fh_cache) >= self._max_open_shards:
            _, old_fh = self._fh_cache.popitem(last=False)
            old_fh.close()

        shard_path = osp.join(self.ROOT, f"tartanair-{shard_idx:06d}.tar")
        fh = open(shard_path, "rb")
        self._fh_cache[shard_idx] = fh
        return fh

    def warmup_fh_cache(self):
        """Pre-open file handles for all shards. Call from worker_init_fn
        to avoid cold-start latency on first access after fork."""
        for shard_idx in sorted(self._shard_offsets.keys()):
            self._get_fh(shard_idx)

    def _read_bytes(self, view_idx, ext):
        """Read raw bytes for a tar member using byte-offset direct seek."""
        shard_idx = self._local_to_shard[view_idx]
        key = self._local_to_key[view_idx]
        member_name = f"{key}.{ext}"

        offsets = self._get_shard_offsets(shard_idx)
        data_offset, data_size = offsets[member_name]

        fh = self._get_fh(shard_idx)
        fh.seek(data_offset)
        return fh.read(data_size)

    _EXTENSIONS = ("rgb.png", "depth.npy.zl", "cam.npz")
    _COALESCE_GAP = 4096  # merge reads within 4 KB gap

    def _prefetch_bytes(self, view_idxs):
        """Pre-fetch all raw bytes for a batch of views with coalesced I/O.

        Groups reads by shard, sorts by offset, and merges adjacent reads
        (within _COALESCE_GAP bytes) into single sequential reads.
        Converts ~72 random seeks into ~1-3 large sequential reads on Lustre.

        Returns: dict mapping (view_idx, ext) -> raw bytes
        """
        # 1. Collect all read requests
        requests = []
        for view_idx in view_idxs:
            shard_idx = self._local_to_shard[view_idx]
            key = self._local_to_key[view_idx]
            offsets = self._get_shard_offsets(shard_idx)
            for ext in self._EXTENSIONS:
                member_name = f"{key}.{ext}"
                data_offset, data_size = offsets[member_name]
                requests.append((shard_idx, data_offset, data_size, view_idx, ext))

        # 2. Group by shard
        by_shard = {}
        for req in requests:
            by_shard.setdefault(req[0], []).append(req)

        result = {}
        for shard_idx, shard_reqs in by_shard.items():
            # 3. Sort by offset for sequential access
            shard_reqs.sort(key=lambda r: r[1])

            # 4. Coalesce adjacent reads
            groups = [[shard_reqs[0]]]
            for req in shard_reqs[1:]:
                prev = groups[-1][-1]
                prev_end = prev[1] + prev[2]  # offset + size
                if req[1] - prev_end <= self._COALESCE_GAP:
                    groups[-1].append(req)
                else:
                    groups.append([req])

            # 5. One read per coalesced group
            fh = self._get_fh(shard_idx)
            for group in groups:
                start = group[0][1]
                end = group[-1][1] + group[-1][2]
                fh.seek(start)
                chunk = fh.read(end - start)
                for _, data_offset, data_size, view_idx, ext in group:
                    lo = data_offset - start
                    result[(view_idx, ext)] = chunk[lo:lo + data_size]

        return result

    def _tar_imread_cv2(self, view_idx, prefetched=None):
        """imread_cv2 equivalent: decode PNG image from tar shard."""
        raw = prefetched[(view_idx, "rgb.png")] if prefetched else self._read_bytes(view_idx, "rgb.png")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            key = self._local_to_key[view_idx]
            raise IOError(f"Could not load image={key}.rgb.png")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _tar_np_load(self, view_idx, ext, prefetched=None):
        """np.load equivalent: load .npy / .npz from tar shard.
        Supports zlib-compressed variants (.npy.zl)."""
        raw = prefetched[(view_idx, ext)] if prefetched else self._read_bytes(view_idx, ext)
        if ext.endswith(".zl"):
            raw = zlib.decompress(raw)
        return np.load(io.BytesIO(raw))

    # ------------------------------------------------------------------
    # Public interface (mirrors TartanAir_Multi)
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=0.8,
            fix_interval_prob=0.8,
            block_shuffle=16,
        )
        image_idxs = np.array(all_image_ids)[pos]

        # Pre-fetch all bytes with coalesced I/O (72 seeks -> ~1-3 reads)
        prefetched = self._prefetch_bytes(image_idxs)

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = self.scenes[scene_id]
            basename = self.images[view_idx]

            img = basename + "_rgb.png"
            image = self._tar_imread_cv2(view_idx, prefetched)
            depthmap = self._tar_np_load(view_idx, "depth.npy.zl", prefetched)
            camera_params = self._tar_np_load(view_idx, "cam.npz", prefetched)

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= 1000
            depthmap[sky_mask] = -1.0  # sky
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[1.0, 0.0, 0.0]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="TartanAir",
                    label=scene_dir,
                    is_metric=self.is_metric,
                    instance=scene_dir + "_" + img,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
