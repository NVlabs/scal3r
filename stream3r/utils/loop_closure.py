# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Online loop closure detection using DINOv2-B + SALAD (8448-dim) + FAISS.

Uses the same VPR model as VGGT-Long for high-quality place recognition.

Usage:
    detector = OnlineLoopDetector(device='cuda')
    for frame_idx in keyframe_indices:
        loops = detector.add_and_query(frame_idx, image_paths[frame_idx])
        # loops: [(past_frame_idx, similarity_score), ...]
    detector.reset()  # call between sequences
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Tuple

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# Path to VGGT-Long reference code
_VGGT_LONG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reference", "VGGT-Long"
)


class OnlineLoopDetector:
    """Incremental loop detection using DINOv2 ViT-B + SALAD aggregator + FAISS.

    Descriptor: 8448-dim (64 clusters * 128 dim + 256 token dim), L2-normalized.
    Same architecture as VGGT-Long for high-quality place recognition.
    """

    def __init__(
        self,
        device: str = "cuda",
        similarity_threshold: float = 0.85,
        temporal_gap: int = 10,
        max_loops_per_frame: int = 1,
        nms_window: int = 25,
    ):
        if not _HAS_FAISS:
            raise ImportError("faiss is required for loop closure. Install with: pip install faiss-cpu")

        self.device = device
        self.similarity_threshold = similarity_threshold
        self.temporal_gap = temporal_gap
        self.max_loops_per_frame = max_loops_per_frame
        self.nms_window = nms_window
        self._recent_loop_targets: List[Tuple[int, int]] = []

        # Incremental FAISS index (inner product on L2-normalized vectors = cosine sim)
        self._descriptor_dim = 64 * 128 + 256  # 8448

        # Build DINOv2-B + SALAD model (same as VGGT-Long)
        self.model = self._build_vpr_model()
        self.model = self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.transform = T.Compose([
            T.Resize((336, 336), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.index = faiss.IndexFlatIP(self._descriptor_dim)
        self.frame_ids: List[int] = []

    def _build_vpr_model(self):
        """Build DINOv2-B + SALAD VPR model using VGGT-Long code."""
        # Temporarily add VGGT-Long to sys.path for imports
        added = False
        if _VGGT_LONG_ROOT not in sys.path:
            sys.path.insert(0, _VGGT_LONG_ROOT)
            added = True
        # DINOv2 backbone uses relative path './LoopModels/dinov2' in its source,
        # so we must temporarily chdir to VGGT-Long root.
        old_cwd = os.getcwd()
        try:
            os.chdir(_VGGT_LONG_ROOT)
            from LoopModels.vpr_model import VPRModel

            dinov2_weights = os.path.join(_VGGT_LONG_ROOT, "weights", "dinov2_vitb14_pretrain.pth")
            salad_weights = os.path.join(_VGGT_LONG_ROOT, "weights", "dino_salad.ckpt")

            if not os.path.isfile(salad_weights):
                raise FileNotFoundError(
                    f"SALAD weights not found at {salad_weights}. "
                    f"Download from VGGT-Long and place in reference/VGGT-Long/weights/"
                )

            config = {
                'Weights': {
                    'SALAD': salad_weights,
                    'DNIO': dinov2_weights,
                }
            }
            model = VPRModel(
                backbone_arch='dinov2_vitb14',
                backbone_config={
                    'num_trainable_blocks': 4,
                    'return_token': True,
                    'norm_layer': True,
                },
                agg_arch='SALAD',
                agg_config={
                    'num_channels': 768,
                    'num_clusters': 64,
                    'cluster_dim': 128,
                    'token_dim': 256,
                },
                vggt_long_config=config,
            )
            model.load_state_dict(torch.load(salad_weights, map_location='cpu'))
            print(f"Loaded VPR model (DINOv2-B + SALAD, {self._descriptor_dim}-dim)")
            return model
        finally:
            os.chdir(old_cwd)
            if added:
                sys.path.remove(_VGGT_LONG_ROOT)

    def reset(self):
        """Reset FAISS index for a new sequence."""
        self.index.reset()
        self.frame_ids.clear()
        self._recent_loop_targets.clear()

    # STream3R uses mean=0.5,std=0.5; DINOv2 uses ImageNet normalization
    _MODEL_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    _MODEL_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    _DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_descriptor(self, image_path: str) -> np.ndarray:
        """Extract SALAD descriptor from a single image (disk path).
        SALAD output is already L2-normalized internally.

        Returns: (1, 8448) float32 numpy array.
        """
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feat = self.model(x)  # (1, 8448), already L2-normalized by SALAD
        return feat.float().cpu().numpy()

    @torch.no_grad()
    def extract_descriptor_from_tensor(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Extract SALAD descriptor from a model-normalized GPU tensor.

        Args:
            img_tensor: (1, 3, H, W) tensor, normalized with mean=0.5, std=0.5

        Returns: (1, 8448) float32 numpy array.
        """
        # Squeeze extra dims: [B, S, 3, H, W] or [1, 1, 3, H, W] → [1, 3, H, W]
        while img_tensor.dim() > 4:
            img_tensor = img_tensor.squeeze(0) if img_tensor.shape[0] == 1 else img_tensor.squeeze(1)
        # Reverse model normalization → [0, 1]
        mean_m = self._MODEL_MEAN.to(img_tensor.device)
        std_m = self._MODEL_STD.to(img_tensor.device)
        x = img_tensor * std_m + mean_m
        # Resize to 336x336 for DINOv2
        x = F.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False)
        # Apply DINOv2 ImageNet normalization
        mean_d = self._DINO_MEAN.to(x.device)
        std_d = self._DINO_STD.to(x.device)
        x = (x - mean_d) / std_d
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feat = self.model(x)
        return feat.float().cpu().numpy()

    def add_and_query(
        self, frame_idx: int, image_path: str = None, add_to_index: bool = True,
        img_tensor: torch.Tensor = None,
    ) -> List[Tuple[int, float]]:
        """Query for loop closure candidates, optionally add descriptor to index.

        Args:
            frame_idx: current frame index
            image_path: path to current frame image (fallback if no img_tensor)
            add_to_index: if True, add descriptor to FAISS index (only for KFs)
            img_tensor: (1, 3, H, W) GPU tensor, avoids disk I/O if provided

        Returns: list of (past_frame_idx, similarity_score), or empty.
        """
        if img_tensor is not None:
            desc = self.extract_descriptor_from_tensor(img_tensor)
        else:
            desc = self.extract_descriptor(image_path)  # (1, 8448)

        # Query existing index (KF descriptors only)
        loops = []
        if self.index.ntotal > 0:
            k = min(5, self.index.ntotal)
            scores, indices = self.index.search(desc, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                past_frame = self.frame_ids[idx]
                if (score > self.similarity_threshold
                        and abs(frame_idx - past_frame) > self.temporal_gap):
                    # NMS: skip if a recent loop already targeted a nearby frame
                    suppressed = False
                    for q, t in self._recent_loop_targets:
                        if (abs(frame_idx - q) < self.nms_window
                                and abs(past_frame - t) < self.nms_window):
                            suppressed = True
                            break
                    if not suppressed:
                        loops.append((past_frame, float(score)))
            loops.sort(key=lambda x: -x[1])
            loops = loops[:self.max_loops_per_frame]

        # Debug: log top score + best distant match
        if self.index.ntotal > 0 and len(self.frame_ids) > 0:
            top_score = float(scores[0, 0]) if self.index.ntotal > 0 else 0
            top_past = self.frame_ids[indices[0, 0]] if indices[0, 0] >= 0 else -1
            gap = abs(frame_idx - top_past) if top_past >= 0 else 0
            # Find best distant match (gap > temporal_gap) among all k results
            best_distant_score, best_distant_frame = 0.0, -1
            for s, ii in zip(scores[0], indices[0]):
                if ii < 0:
                    continue
                pf = self.frame_ids[ii]
                if abs(frame_idx - pf) > self.temporal_gap and s > best_distant_score:
                    best_distant_score = float(s)
                    best_distant_frame = pf
            if frame_idx % 100 == 0:
                distant_str = f", best_distant={best_distant_frame}(score={best_distant_score:.4f})" if best_distant_frame >= 0 else ""
                print(f"  [Loop debug] frame {frame_idx}: top_score={top_score:.4f}, "
                      f"top_match={top_past} (gap={gap}), index_size={self.index.ntotal}, k={k}{distant_str}")

        # Record accepted loops for NMS
        for past_frame, _ in loops:
            self._recent_loop_targets.append((frame_idx, past_frame))

        # Only add KF descriptors to index
        if add_to_index:
            self.index.add(desc)
            self.frame_ids.append(frame_idx)

        return loops


class GTLoopDetector:
    """GT-based loop detector using ground truth poses. For ablation only.

    Finds loop pairs where GT spatial distance < dist_threshold.
    Same interface as OnlineLoopDetector (add_and_query / reset).
    """

    def __init__(
        self,
        gt_poses_file: str,
        dist_threshold: float = 5.0,
        temporal_gap: int = 300,
        max_loops_per_frame: int = 1,
        nms_window: int = 50,
        reset_interval: int = 1000000,
        **kwargs,
    ):
        poses = np.loadtxt(gt_poses_file)
        # Extract (tx, tz) from 3x4 row-major: columns 3 and 11
        orig_positions = poses[:, [3, 11]]
        # Build view-indexed positions (account for overlap frames from reset_interval)
        view_positions = []
        for i in range(len(orig_positions)):
            view_positions.append(orig_positions[i])
            if (i + 1) % reset_interval == 0:
                view_positions.append(orig_positions[i])  # overlap frame
        self._positions = np.array(view_positions)
        self.dist_threshold = dist_threshold
        self.temporal_gap = temporal_gap
        self.max_loops_per_frame = max_loops_per_frame
        self.nms_window = nms_window
        self._past_keyframes: List[int] = []
        self._recent_loop_targets: List[Tuple[int, int]] = []

    def reset(self):
        self._past_keyframes.clear()
        self._recent_loop_targets.clear()

    def add_and_query(
        self, frame_idx: int, image_path: str = "", add_to_index: bool = True
    ) -> List[Tuple[int, float]]:
        if frame_idx >= len(self._positions):
            if add_to_index:
                self._past_keyframes.append(frame_idx)
            return []

        pos_curr = self._positions[frame_idx]
        loops = []
        for past_idx in self._past_keyframes:
            if past_idx >= len(self._positions):
                continue
            if abs(frame_idx - past_idx) <= self.temporal_gap:
                continue
            dist = float(np.linalg.norm(pos_curr - self._positions[past_idx]))
            if dist < self.dist_threshold:
                print(f"    GT loop match: view {frame_idx}<->{past_idx}, dist={dist:.1f}m")
                # NMS
                suppressed = False
                for q, t in self._recent_loop_targets:
                    if (abs(frame_idx - q) < self.nms_window
                            and abs(past_idx - t) < self.nms_window):
                        suppressed = True
                        break
                if not suppressed:
                    # Score: closer = higher (1.0 at 0m, 0.0 at dist_threshold)
                    score = 1.0 - dist / self.dist_threshold
                    loops.append((past_idx, score))

        loops.sort(key=lambda x: -x[1])
        loops = loops[:self.max_loops_per_frame]

        for past_idx, _ in loops:
            self._recent_loop_targets.append((frame_idx, past_idx))

        if add_to_index:
            self._past_keyframes.append(frame_idx)
        return loops
