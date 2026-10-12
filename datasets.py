import argparse
import h5py
import numpy as np
import scipy.io as sio
import torch
from pathlib import Path
from torch.utils.data import Dataset
from utils import BaseAugmentation

DATASET_CONFIGS = {
    'pavia': {
        'name': 'Pavia',
        'in_channels': 102,
        'domains': ('PC', 'UP'),
        'files': {
            'PC': ('paviaC.mat', 'paviaC_7gt.mat'),
            'UP': ('paviaU.mat', 'paviaU_7gt.mat'),
        },
        'target_config': {
            'num_bands': 102,
            'spectral_mask_ratio': 0.15,
            'lambda_warm_recon': 0.5,
            'lambda_warm_var': 0.02,
            'lambda_warm_cov': 0.002,
            'source_prototypes_per_class': 3,
            'target_clusters_per_class': 4,
            'knn_k': 10,
            'confidence_threshold': 0.48,
            'initial_coverage': 0.2,
            'final_coverage': 0.85,
            'min_samples_per_class': 16,
            'pseudo_temperature': 0.25,
            'uot_reg': 0.05,
            'uot_reg_m': 0.8,
            'uot_iterations': 300,
            'lambda_cons': 0.25,
            'lambda_src_proto': 0.25,
            'lambda_memory': 0.7,
            'lambda_topology': 0.05,
            'teacher_temperature': 0.7,
            'ema_momentum': 0.999,
            'memory_per_class': 300,
            'memory_batch_size': 64,
        },
    },
    'houston': {
        'name': 'Houston',
        'in_channels': 48,
        'domains': ('H13', 'H18'),
        'files': {
            'H13': ('Houston13.mat', 'Houston13_7gt.mat'),
            'H18': ('Houston18.mat', 'Houston18_7gt.mat'),
        },
        'target_config': {
            'num_bands': 48,
            'spectral_mask_ratio': 0.08,
            'lambda_warm_recon': 0.25,
            'lambda_warm_var': 0.01,
            'lambda_warm_cov': 0.001,
            'source_prototypes_per_class': 2,
            'target_clusters_per_class': 2,
            'knn_k': 5,
            'confidence_threshold': 0.65,
            'initial_coverage': 0.08,
            'final_coverage': 0.45,
            'min_samples_per_class': 8,
            'pseudo_temperature': 0.35,
            'uot_reg': 0.08,
            'uot_reg_m': 0.5,
            'uot_iterations': 200,
            'lambda_cons': 0.1,
            'lambda_src_proto': 0.12,
            'lambda_memory': 0.2,
            'lambda_topology': 0.02,
            'teacher_temperature': 1.0,
            'ema_momentum': 0.99,
            'memory_per_class': 100,
            'memory_batch_size': 32,
        },
    },
}

CLI_DEFAULTS = {
    'source_train': {
        'pavia': {
            'patch_size': 15,
            'gpu': '5',
            'batch_size': 256,
            'lr': 0.0005,
            'max_epoch': 15,
            'prototypes_per_class': 3,
            'label_smoothing': 0.0,
            'rebalance_start_ratio': 0.5,
        },
        'houston': {
            'patch_size': 13,
            'gpu': '6',
            'batch_size': 64,
            'lr': 0.001,
            'label_smoothing': 0.02,
            'max_epoch': 30,
            'prototypes_per_class': 2,
            'rebalance_start_ratio': 0.65,
        },
    },
    'target_adapt': {
        'pavia': {
            'patch_size': 15,
            'max_epoch': 5,
            'warmup_epoch': 2,
            'gpu': '5',
            'batch_size': 128,
            'seed': 100,
            'source_prototypes_per_class': 3,
            'target_clusters_per_class': 4,
            'knn_k': 2,
            'confidence_threshold': 0.01,
            'initial_coverage': 0.01,
            'final_coverage': 0.02,
            'min_samples_per_class': 0,
            'lambda_cons': 0.15,
            'lambda_src_proto': 0.25,
            'memory_per_class': 300,
            'memory_batch_size': 64,
            'ema_momentum': 0.999,
            'teacher_temperature': 0.7,
            'source_prior_correction_tau': -1.0,
        },
        'houston': {
            'patch_size': 13,
            'max_epoch': 6,
            'warmup_epoch': 6,
            'gpu': '4',
            'batch_size': 64,
            'seed': 25,
            'source_prototypes_per_class': 2,
            'target_clusters_per_class': 2,
            'knn_k': 5,
            'confidence_threshold': 0.0,
            'initial_coverage': 0.1,
            'final_coverage': 1.0,
            'min_samples_per_class': 8,
            'source_prior_correction_tau': -1.5,
            'lambda_cons': 0.18,
            'lambda_src_proto': 0.12,
            'memory_per_class': 100,
            'memory_batch_size': 32,
            'ema_momentum': 0.99,
            'teacher_temperature': 1.0,
        },
    },
    'eval': {
        'pavia': {
            'patch_size': 15,
            'gpu': '5',
            'seed': 25,
        },
        'houston': {
            'patch_size': 13,
            'gpu': '2',
            'seed': 100,
        },
    },
}


def get_dataset_config(dataset):
    try:
        return DATASET_CONFIGS[dataset.lower()]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {dataset!r}; use pavia or houston.") from error


def validate_domain_pair(source_domain, target_domain):
    if source_domain == target_domain:
        raise ValueError("source_domain and target_domain must be different.")


class DatasetArgumentParser(argparse.ArgumentParser):
    def __init__(self, phase, **kwargs):
        super().__init__(**kwargs)
        self.phase = phase
        self.add_argument("--dataset", type=str.lower, choices=tuple(DATASET_CONFIGS), default="pavia")

    def parse_known_args(self, args=None, namespace=None):
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--dataset", type=str.lower, choices=tuple(DATASET_CONFIGS), default="pavia")
        selected, _ = probe.parse_known_args(args)
        config = get_dataset_config(selected.dataset)
        defaults = dict(CLI_DEFAULTS[self.phase][selected.dataset])
        defaults.update(
            data_path=f"./{config['name']}",
            save_path=f"./results/{selected.dataset}",
            in_channels=config['in_channels'],
            source_domain=config['domains'][0],
        )
        if self.phase != "source_train":
            defaults["target_domain"] = config['domains'][1]
        if self.phase == "target_adapt":
            defaults["model_path"] = defaults["save_path"]
        self.set_defaults(**defaults)
        return super().parse_known_args(args, namespace)

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        config = get_dataset_config(parsed.dataset)
        for name in ("source_domain", "target_domain"):
            domain = getattr(parsed, name, None)
            if domain is not None and domain not in config['domains']:
                self.error(f"--{name} for {parsed.dataset} must be one of {config['domains']}.")
        if parsed.in_channels != config['in_channels']:
            self.error(f"{parsed.dataset} requires {config['in_channels']} spectral bands.")
        if parsed.patch_size <= 0 or parsed.patch_size % 2 == 0:
            self.error("--patch_size must be a positive odd integer.")
        return parsed


class HSIDataset(Dataset):
    cls_num = 7

    def __init__(self, file_path, domain=None, patch_size=15, filter_func=None,
                 return_id=False, data_augment=False, dataset="pavia",
                 src_mat="Houston13.mat", tar_mat="Houston18.mat",
                 src_gt="Houston13_7gt.mat", tar_gt="Houston18_7gt.mat"):
        config = get_dataset_config(dataset)
        self.dataset = dataset.lower()
        self.in_chs = config['in_channels']
        self.domain = domain or config['domains'][0]
        if self.domain not in config['domains']:
            raise ValueError(f"Invalid {dataset} domain: {self.domain!r}.")
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer.")
        self.patch_size = int(patch_size)
        self.filter_func = filter_func
        self.return_id = return_id
        self.data_augment = data_augment
        self.base_aug = BaseAugmentation() if data_augment else None
        files = config['files'] if self.dataset == "pavia" else {
            "H13": (src_mat, src_gt), "H18": (tar_mat, tar_gt),
        }
        data_name, label_name = files[self.domain]
        data_path, label_path = Path(file_path) / data_name, Path(file_path) / label_name
        for path in (data_path, label_path):
            if not path.exists():
                raise FileNotFoundError(f"{dataset} file not found: {path}")
        if self.dataset == "houston":
            with h5py.File(data_path, "r") as handle:
                data = handle["ori_data"][:]
            with h5py.File(label_path, "r") as handle:
                label_map = handle["map"][:]

            # h5py reverses the axes of MATLAB v7.3 arrays.
            self.data = np.transpose(data, (0, 2, 1)).astype(np.float32)
            self.label_map = np.asarray(label_map.T)
        else:
            data = sio.loadmat(data_path)["ori_data"]
            self.data = np.transpose(data, (2, 0, 1)).astype(np.float32)
            self.label_map = sio.loadmat(label_path)["map"]
        if self.data.shape[0] != self.in_chs:
            raise ValueError(f"{dataset} requires {self.in_chs} bands, got {self.data.shape}.")
        if self.data.shape[1:] != self.label_map.shape:
            raise ValueError(f"Data/label shapes do not match: {self.data.shape}, {self.label_map.shape}.")
        bmin = self.data.min(axis=(1, 2), keepdims=True)
        bmax = self.data.max(axis=(1, 2), keepdims=True)
        self.data = (self.data - bmin) / np.where(bmax - bmin > 1e-8, bmax - bmin, 1.0)
        margin = self.patch_size // 2
        self.padded_data = np.pad(self.data, ((0, 0), (margin, margin), (margin, margin)), mode="reflect")
        valid_mask = self.label_map > 0
        self.coords = np.stack(np.where(valid_mask), axis=-1)
        self.labels = self.label_map[valid_mask].astype(np.int64) - 1
        if len(self.labels) and self.labels.max() >= self.cls_num:
            raise ValueError(f"Labels must be in MATLAB range 0-{self.cls_num}.")
        if filter_func is not None:
            self.apply_filter()

    def apply_filter(self):
        mask = np.asarray([self.filter_func(label) for label in self.labels], dtype=bool)
        self.coords = self.coords[mask]
        self.labels = self.labels[mask]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row, col = self.coords[index]
        patch = self.padded_data[:, row:row + self.patch_size, col:col + self.patch_size]
        image = torch.from_numpy(patch.copy()).float()
        if self.base_aug is not None:
            image = self.base_aug(image)
        label = self.labels[index]
        return (image, label, index) if self.return_id else (image, label)
