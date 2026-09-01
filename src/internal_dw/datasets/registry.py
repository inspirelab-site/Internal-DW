import os
from typing import Dict, Type

from torch.utils.data import DataLoader, DistributedSampler

from .base import SequenceDataset, sequence_collate
from .hcp import HCPMovieDataset, build_hcp_splits
from .gait import GaitDataset, build_gait_splits
from .prepared_temporal import (
    PreparedTemporalAutonomousDataset,
    PreparedTemporalDrivenDataset,
    build_prepared_temporal_splits,
)
from .ieeg import IEEGDataset, build_ieeg_splits
from .lorenz96 import Lorenz96Dataset, build_lorenz96_splits
from .synthetic_memory import (
    DrivenMackeyGlassDataset,
    KnownSNRARDataset,
    MackeyGlassDataset,
    NARMADataset,
    build_driven_mackey_glass_splits,
    build_known_snr_ar_splits,
    build_mackey_glass_splits,
    build_narma_splits,
)
from .thewell import TheWell2DDataset, build_thewell_splits
from .weatherbench import WeatherBenchDataset, build_weatherbench_splits
from .weatherbench2 import WeatherBench2Dataset, build_weatherbench2_splits
from .sevir import SEVIRVILDataset, build_sevir_splits
from .kth_actions import KTHActionsDataset, build_kth_actions_splits

_DATASETS: Dict[str, Type[SequenceDataset]] = {
    "gait": GaitDataset,
    "hcp_movie": HCPMovieDataset,
    "ieeg": IEEGDataset,
    "known_snr_ar": KnownSNRARDataset,
    "lorenz96": Lorenz96Dataset,
    "mackey_glass": MackeyGlassDataset,
    "mackey_glass_driven": DrivenMackeyGlassDataset,
    "narma": NARMADataset,
    "prepared_temporal_autonomous": PreparedTemporalAutonomousDataset,
    "prepared_temporal_driven": PreparedTemporalDrivenDataset,
    "the_well": TheWell2DDataset,
    "weatherbench": WeatherBenchDataset,
    "weatherbench2": WeatherBench2Dataset,
    "sevir": SEVIRVILDataset,
    "kth_actions": KTHActionsDataset,
}


def list_datasets():
    return sorted(_DATASETS.keys())


def dataset_has_external_input(name: str) -> bool:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset={name}. Available: {list_datasets()}")
    return bool(getattr(_DATASETS[name], "has_external_input", False))


def dataset_task_type(name: str) -> str:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset={name}. Available: {list_datasets()}")
    return str(getattr(_DATASETS[name], "task_type", "generic_sequence"))


def dataset_evaluator_name(name: str) -> str:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset={name}. Available: {list_datasets()}")
    return str(getattr(_DATASETS[name], "evaluator_name", "generic"))


def build_dataset(name: str, **kwargs) -> SequenceDataset:
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset={name}. Available: {list_datasets()}")
    return _DATASETS[name](**kwargs)


def build_dataloaders(args, rank: int = 0, world_size: int = 1):
    if args.dataset == "hcp_movie":
        train_set, val_set, test_set = build_hcp_splits(args)
    elif args.dataset == "lorenz96":
        train_set, val_set, test_set = build_lorenz96_splits(args)
    elif args.dataset == "gait":
        train_set, val_set, test_set = build_gait_splits(args)
    elif args.dataset == "ieeg":
        train_set, val_set, test_set = build_ieeg_splits(args)
    elif args.dataset == "known_snr_ar":
        train_set, val_set, test_set = build_known_snr_ar_splits(args)
    elif args.dataset == "mackey_glass":
        train_set, val_set, test_set = build_mackey_glass_splits(args)
    elif args.dataset == "mackey_glass_driven":
        train_set, val_set, test_set = build_driven_mackey_glass_splits(args)
    elif args.dataset == "narma":
        train_set, val_set, test_set = build_narma_splits(args)
    elif args.dataset == "prepared_temporal_autonomous":
        train_set, val_set, test_set = build_prepared_temporal_splits(
            args, expect_external=False
        )
    elif args.dataset == "prepared_temporal_driven":
        train_set, val_set, test_set = build_prepared_temporal_splits(
            args, expect_external=True
        )
    elif args.dataset == "the_well":
        train_set, val_set, test_set = build_thewell_splits(args)
    elif args.dataset == "weatherbench":
        train_set, val_set, test_set = build_weatherbench_splits(args)
    elif args.dataset == "weatherbench2":
        train_set, val_set, test_set = build_weatherbench2_splits(args)
    elif args.dataset == "sevir":
        train_set, val_set, test_set = build_sevir_splits(args)
    elif args.dataset == "kth_actions":
        train_set, val_set, test_set = build_kth_actions_splits(args)
    else:
        raise ValueError(f"Unknown dataset={args.dataset}. Available: {list_datasets()}")

    def make_loader(ds, shuffle):
        # WeatherBench-2 pilot validation/test splits contain only about 22
        # non-overlapping segments.  Sharding them would make rank 0 select a
        # checkpoint from roughly one quarter of the validation year.  Train is
        # still sharded; evaluation is replicated so every rank (and therefore
        # rank 0's early-stopping decision) sees the complete split.
        replicate_eval = args.dataset == "weatherbench2" and not shuffle
        sampler = (
            DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
            if world_size > 1 and not replicate_eval else None
        )
        nw = int(args.num_workers)
        # Persistent workers + deeper prefetch keep the (few, high-latency on a
        # network mount) file reads overlapped with GPU compute and avoid tearing
        # down / rebuilding worker processes -- and reopening NAS file handles --
        # every epoch. Only valid when num_workers>0.
        extra = dict(persistent_workers=True,
                     prefetch_factor=int(os.environ.get("PREFETCH_FACTOR", "4"))) if nw > 0 else {}
        return DataLoader(
            ds,
            batch_size=args.local_batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=nw,
            pin_memory=True,
            drop_last=False,
            collate_fn=sequence_collate,
            **extra,
        )

    return make_loader(train_set, True), make_loader(val_set, False), make_loader(test_set, False)
