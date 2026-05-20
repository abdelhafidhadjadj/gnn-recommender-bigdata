"""CUDA / device detection — mirrors the notebook's multi-GPU branching."""
from __future__ import annotations
import os
from dataclasses import dataclass
import torch


@dataclass
class DeviceConfig:
    num_gpus: int
    device: torch.device
    embed_device: torch.device   # SBERT / FAISS encoding
    sage_device: torch.device
    gat_device: torch.device
    use_ddp: bool


def detect_devices() -> DeviceConfig:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    num_gpus = torch.cuda.device_count()

    if num_gpus >= 2:
        return DeviceConfig(
            num_gpus=num_gpus,
            device=torch.device("cuda:0"),
            embed_device=torch.device("cuda:0"),
            sage_device=torch.device("cuda:0"),
            gat_device=torch.device("cuda:1"),
            use_ddp=True,
        )
    elif num_gpus == 1:
        gpu = torch.device("cuda:0")
        return DeviceConfig(
            num_gpus=1,
            device=gpu,
            embed_device=gpu,
            sage_device=gpu,
            gat_device=gpu,
            use_ddp=False,
        )
    else:
        cpu = torch.device("cpu")
        return DeviceConfig(
            num_gpus=0,
            device=cpu,
            embed_device=cpu,
            sage_device=cpu,
            gat_device=cpu,
            use_ddp=False,
        )
