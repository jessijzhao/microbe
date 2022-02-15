import os
import pickle
import shutil
from typing import Any, Dict, List

import pytest
import torch
from helpers import get_n_byte_tensor, skipifnocuda

from utils import device, get_layer_set, get_path, reset_peak_memory_stats, save_results


@pytest.mark.parametrize(
    "layer_set, layers",
    [
        # Pytorch layers are named layer (no DP) or gsm_layer (DP)
        ("linear", ["linear", "gsm_linear"]),
        ("conv", ["conv", "gsm_conv"]),
        # Opacus layers are named dplayer (no DP) or gsm_dplayer (DP)
        ("mha", ["mha", "dpmha", "gsm_dpmha"]),
        # RNN-based models share the same interface
        ("rnn_base", ["rnn", "dprnn", "gsm_dprnn"]),
        ("rnn_base", ["lstm", "dplstm", "gsm_dplstm"]),
    ],
)
def test_get_layer_set(layer_set: str, layers: List[str]) -> None:
    """Tests assignment of individual layers to the layer set.

    Args:
        layer_set: layer set (e.g. linear, rnn_base)
        layers: non-exhaustive list of layers that belong to the layer_set
    """
    assert all(get_layer_set(layer) == layer_set for layer in layers)


@skipifnocuda
@pytest.mark.parametrize(
    "prev_max_memory, allocated_memory",
    [
        # prev_max_memory = allocated_memory = 0 --> (0, 0)
        (0, 0),
        # prev_max_memory = allocated_memory > 0 --> (prev_max_memory, prev_max_memory)
        (1, 1),
        # prev_max_memory > allocated_memory = 0 --> (prev_max_memory, 0)
        (1, 0),
        # prev_max_memory > allocated_memory > 0 --> (prev_max_memory, allocated_memory)
        (2, 1),
    ],
)
def test_reset_peak_memory_stats(prev_max_memory: int, allocated_memory: int) -> None:
    """Tests resetting of peak memory stats.

    Notes: Only the relative and not the absolute sizes of prev_max_memory and
    allocated_memory are relevant.

    Args:
        prev_max_memory: current maximum memory stat to simulate
        allocated_memory: current allocated memory to simulate
    """
    # keep x, delete y
    x = get_n_byte_tensor(allocated_memory)
    y = get_n_byte_tensor(prev_max_memory - allocated_memory)
    del y

    # get the true allocated memory (CUDA memory is allocated in blocks)
    prev_max_memory = torch.cuda.max_memory_allocated(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    assert prev_max_memory >= allocated_memory
    assert reset_peak_memory_stats(device) == (prev_max_memory, allocated_memory)

    # clean up
    del x
    torch.cuda.reset_peak_memory_stats(device)
    assert torch.cuda.max_memory_allocated(device) == 0
    assert torch.cuda.memory_allocated(device) == 0


@pytest.mark.parametrize(
    "config, path",
    [
        (
            {"layer": "linear", "batch_size": 64, "num_runs": 10, "num_repeats": 100},
            "./results/raw/linear_bs_64_runs_10_repeats_100.pkl",
        ),
        (
            {"layer": "gsm_rnn", "batch_size": 128, "num_runs": 5, "num_repeats": 20},
            "./results/raw/gsm_rnn_bs_128_runs_5_repeats_20.pkl",
        ),
    ],
)
def test_get_path(config: Dict[str, Any], path: str) -> None:
    """Tests result pickle path generation.

    Args:
        config: arguments to pass to get_path
        path: corresponding path
    """
    assert path == get_path(**config)


@pytest.mark.parametrize(
    "config, root",
    [
        (
            {
                "layer": "linear",
                "batch_size": 64,
                "num_runs": 10,
                "num_repeats": 100,
                "results": [],
                "config": {},
            },
            "tests/tmp/",
        )
    ],
)
def test_save_results(config: Dict[str, Any], root: str) -> None:
    """Tests saving benchmark results.

    Args:
        config: arguments to pass to save_results
        root: directory to temporarily store results
    """
    os.mkdir(root)
    save_results(**config, root=root)

    with open(
        get_path(
            layer=config["layer"],
            batch_size=config["batch_size"],
            num_runs=config["num_runs"],
            num_repeats=config["num_repeats"],
            root=root,
        ),
        "rb",
    ) as f:
        data = pickle.load(f)
        for key, value in config.items():
            assert data[key] == value

    shutil.rmtree(root)