# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .bedlam import BEDLAM_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .cop3d import Cop3D_Multi
from .dl3dv import DL3DV_Multi
from .dynamic_replica import DynamicReplica
from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .hoi4d import HOI4D_Multi
from .irs import IRS
from .mapfree import MapFree_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mp3d import MP3D_Multi
from .mvimgnet import MVImgNet_Multi
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .realestate10k import RE10K_Multi
from .scannet import ScanNet_Multi
from .scannetpp import ScanNetpp_Multi  # noqa
from .smartportraits import SmartPortraits_Multi
from .spring import Spring
from .synscapes import SynScapes
from .tartanair import TartanAir_Multi
from .tartanair_wds import TartanAirWds_Multi
from .tartanair_h5 import TartanAirH5_Multi
from .threedkb import ThreeDKenBurns
from .uasol import UASOL_Multi
from .urbansyn import UrbanSyn
from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa


from accelerate import Accelerator


def _find_warmup_datasets(dataset):
    """Recursively find all datasets needing warmup (WDS and H5) through wrappers."""
    if isinstance(dataset, (TartanAirWds_Multi, TartanAirH5_Multi)):
        yield dataset
    elif hasattr(dataset, 'dataset'):  # ResizedDataset, MulDataset
        yield from _find_warmup_datasets(dataset.dataset)
    elif hasattr(dataset, 'datasets'):  # CatDataset
        for ds in dataset.datasets:
            yield from _find_warmup_datasets(ds)


def _make_warmup_worker_init_fn(dataset):
    """Create a worker_init_fn that pre-opens file handles after fork."""
    warmup_datasets = list(_find_warmup_datasets(dataset))
    if not warmup_datasets:
        return None

    def worker_init_fn(worker_id):
        for ds in warmup_datasets:
            ds.warmup_fh_cache()

    return worker_init_fn


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    worker_init_fn = _make_warmup_worker_init_fn(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=num_workers > 0,
            prefetch_factor=8 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=num_workers > 0,
            prefetch_factor=8 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    return data_loader
