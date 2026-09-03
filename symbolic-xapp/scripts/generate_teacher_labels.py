"""Generate clean frozen adaptive-K RL-teacher labels for WP-MEC.

Released workflow for the single N=10 reproducibility point
------------------------------------------------------------
1. Use only the first 24,000 chronological development frames for online
   MemoryDNN pre-training.
2. Freeze the N-120-80-N network and save a complete teacher checkpoint.
3. Reset adaptive K to N and, without any encode/learn call, generate labels
   for every channel frame.
4. Validate every mode/allocation, verify that the frozen network hash did not
   change, and atomically publish the MAT/JSON outputs.

The script is resumable within the new experiment directory.  It never reads
old labels/models/results and never overwrites a completed formal result unless
``--force`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# Set before NumPy/SciPy/PyTorch imports.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import scipy
import scipy.io as sio
import torch

from symbolic_xapp.memory_dnn import MemoryDNN, set_deterministic_seed
from symbolic_xapp.resource_allocator import DEFAULT_BISECTION_DELTA, bisection


PROTOCOL_NAME = "paper-v6-clean-fit-validation-test"
GENERATOR_VERSION = "frozen-adaptive-k-teacher-v6"
DEFAULT_USERS = (10,)
PRETRAIN_FRAMES = 24_000
LEARNING_RATE = 0.01
TRAINING_INTERVAL = 10
BATCH_SIZE = 128
MEMORY_SIZE = 1024
ADAPTIVE_K_INTERVAL = 32
RANDOM_SEED = 42
CHANNEL_SCALE = 1_000_000.0
CHECKPOINT_INTERVAL = 500
OPTIMIZER_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1.0e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def model_parameter_sha256(memory_dnn: MemoryDNN) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(memory_dnn.model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def atomic_savemat(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mat")
    sio.savemat(str(temporary), payload, do_compression=True, appendmat=False)
    os.replace(str(temporary), str(path))


def torch_load_complete(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def save_memory_checkpoint_atomic(
    memory_dnn: MemoryDNN,
    path: Path,
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    memory_dnn.save_checkpoint(temporary, extra_metadata=metadata)
    os.replace(str(temporary), str(path))


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(str(temporary), **arrays)
    os.replace(str(temporary), str(path))


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    set_deterministic_seed(int(seed), deterministic=True)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def environment_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def load_protocol(root: Path) -> Dict[str, Any]:
    path = root / "configs" / "experiment_protocol.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"experiment protocol not found: {path}. Run 00_initialize_experiment.py first."
        )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError(
            f"unexpected protocol {protocol.get('protocol_name')!r}; "
            f"expected {PROTOCOL_NAME!r}"
        )
    return protocol


def load_channels(path: Path, n_users: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"channel file not found: {path}")
    data = sio.loadmat(str(path), variable_names=["input_h"])
    if "input_h" not in data:
        raise KeyError(f"{path} does not contain 'input_h'")
    channels = np.asarray(data["input_h"], dtype=np.float64)
    if channels.ndim != 2:
        raise ValueError(f"input_h must be two-dimensional, got {channels.shape}")
    if channels.shape[1] == n_users:
        pass
    elif channels.shape[0] == n_users:
        channels = channels.T
    else:
        raise ValueError(
            f"input_h shape {channels.shape} is incompatible with N={n_users}"
        )
    if channels.shape[0] == 0:
        raise ValueError(f"no channel frames found in {path}")
    if not np.all(np.isfinite(channels)):
        raise ValueError(f"non-finite channel value found in {path}")
    if np.any(channels <= 0.0):
        raise ValueError(f"channel gains must be strictly positive in {path}")
    return np.ascontiguousarray(channels, dtype=np.float64)


def split_indices(num_frames: int) -> Dict[str, int]:
    development_end = int(0.8 * int(num_frames))
    fit_end = int(0.75 * development_end)
    return {
        "fit_end": fit_end,
        "development_end": development_end,
        "num_fit": fit_end,
        "num_validation": development_end - fit_end,
        "num_test": int(num_frames) - development_end,
    }


def update_adaptive_k(
    selected_rank_history: Sequence[int],
    n_users: int,
    update_interval: int,
) -> int:
    if int(update_interval) <= 0:
        raise ValueError("adaptive-K update interval must be positive")
    if len(selected_rank_history) < int(update_interval):
        return int(n_users)
    recent = selected_rank_history[-int(update_interval) :]
    return int(min(max(int(rank) for rank in recent) + 1, int(n_users)))


def make_config(n_users: int, pretrain_frames: int, seed: int) -> Dict[str, Any]:
    return {
        "generator_version": GENERATOR_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "teacher_type": "frozen MemoryDNN RL teacher with adaptive-K candidate generation",
        "network": [int(n_users), 120, 80, int(n_users)],
        "learning_rate": LEARNING_RATE,
        "training_interval": TRAINING_INTERVAL,
        "batch_size": BATCH_SIZE,
        "memory_size": MEMORY_SIZE,
        "adam_betas": list(OPTIMIZER_BETAS),
        "weight_decay": WEIGHT_DECAY,
        "adaptive_k": True,
        "adaptive_k_interval": ADAPTIVE_K_INTERVAL,
        "initial_k": int(n_users),
        "pretrain_frames": int(pretrain_frames),
        "seed": int(seed),
        "quantization": "order-preserving (OP)",
        "channel_scale": CHANNEL_SCALE,
        "bisection_delta": DEFAULT_BISECTION_DELTA,
        "freeze_before_label_generation": True,
        "reset_adaptive_k_before_label_generation": True,
        "label_generation_updates_network": False,
    }


def make_memory_dnn(n_users: int, config: Dict[str, Any]) -> MemoryDNN:
    return MemoryDNN(
        net=config["network"],
        learning_rate=config["learning_rate"],
        training_interval=config["training_interval"],
        batch_size=config["batch_size"],
        memory_size=config["memory_size"],
        seed=config["seed"],
        device="cpu",
        deterministic=True,
        adam_betas=tuple(config["adam_betas"]),
        weight_decay=config["weight_decay"],
    )


def evaluate_candidates(
    channel: np.ndarray,
    candidate_modes: Sequence[Sequence[int]],
    n_users: int,
) -> Tuple[int, np.ndarray, float, float, np.ndarray]:
    if len(candidate_modes) == 0:
        raise RuntimeError("MemoryDNN.decode returned no candidates")

    best_index = -1
    best_mode: Optional[np.ndarray] = None
    best_reward = -np.inf
    best_a = 0.0
    best_tau = np.zeros(n_users, dtype=np.float64)

    for candidate_index, candidate in enumerate(candidate_modes):
        mode = np.asarray(candidate, dtype=np.int8).reshape(-1)
        if mode.shape != (n_users,) or not np.all(np.logical_or(mode == 0, mode == 1)):
            raise ValueError(f"invalid candidate mode: {mode}")
        reward, a_value, _, details = bisection(
            channel,
            mode,
            return_details=True,
            delta=DEFAULT_BISECTION_DELTA,
        )
        tau_full = np.asarray(details["tau_full"], dtype=np.float64)
        if tau_full.shape != (n_users,):
            raise AssertionError(f"tau_full shape is {tau_full.shape}, expected {(n_users,)}")
        if not np.isclose(reward, details["weighted_sum_rate"], rtol=1e-12, atol=1e-6):
            raise AssertionError("allocation objective did not reproduce itself")
        if float(reward) > best_reward:
            best_index = int(candidate_index)
            best_mode = mode.copy()
            best_reward = float(reward)
            best_a = float(a_value)
            best_tau = tau_full.copy()

    if best_mode is None or best_index < 0:
        raise RuntimeError("candidate evaluation did not select an action")
    return best_index, best_mode, best_reward, best_a, best_tau


def pretrain_teacher(
    memory_dnn: MemoryDNN,
    channels: np.ndarray,
    n_users: int,
    config: Dict[str, Any],
    resume_path: Path,
    resume: bool,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    required_frames = int(config["pretrain_frames"])
    if channels.shape[0] < required_frames:
        raise ValueError(
            f"N={n_users}: pretraining needs {required_frames} frames, "
            f"but only {channels.shape[0]} are available"
        )

    selected_ranks: List[int] = []
    rewards = np.empty(required_frames, dtype=np.float64)
    k_history = np.empty(required_frames, dtype=np.int16)
    completed = 0
    elapsed_previous = 0.0

    if resume and resume_path.is_file():
        payload = torch_load_complete(resume_path)
        metadata = dict(payload.get("extra_metadata", {}))
        if metadata.get("stage") != "pretraining":
            raise ValueError(f"invalid pretraining resume stage in {resume_path}")
        if int(metadata.get("n_users", -1)) != n_users:
            raise ValueError(f"resume checkpoint N mismatch in {resume_path}")
        if metadata.get("config") != config:
            raise ValueError(f"resume checkpoint configuration mismatch in {resume_path}")
        restored = MemoryDNN.load_checkpoint(resume_path, map_location="cpu")
        if restored.frozen:
            restored.unfreeze()
        memory_dnn = restored
        completed = int(metadata["completed_frames"])
        selected_ranks = [int(value) for value in metadata["selected_ranks"]]
        rewards[:completed] = np.asarray(metadata["rewards"], dtype=np.float64)
        k_history[:completed] = np.asarray(metadata["k_history"], dtype=np.int16)
        elapsed_previous = float(metadata.get("elapsed_seconds", 0.0))
        if memory_dnn.memory_counter != completed:
            raise ValueError(
                "pretraining checkpoint is internally inconsistent: "
                f"memory_counter={memory_dnn.memory_counter}, completed={completed}"
            )
        print(f"  resuming pre-training from {completed}/{required_frames}")

    current_k = n_users if completed == 0 else int(k_history[completed - 1])
    run_start = time.perf_counter()
    for frame_index in range(completed, required_frames):
        if frame_index > 0 and frame_index % int(config["adaptive_k_interval"]) == 0:
            current_k = update_adaptive_k(
                selected_ranks, n_users, int(config["adaptive_k_interval"])
            )

        channel = channels[frame_index]
        scaled_channel = channel * float(config["channel_scale"])
        candidates = memory_dnn.decode(scaled_channel, current_k, mode="OP")
        best_index, best_mode, reward, _, _ = evaluate_candidates(
            channel, candidates, n_users
        )
        memory_dnn.encode(scaled_channel, best_mode)

        selected_ranks.append(best_index + 1)
        rewards[frame_index] = reward
        k_history[frame_index] = current_k
        completed_now = frame_index + 1

        if (
            completed_now % int(checkpoint_interval) == 0
            or completed_now == required_frames
        ):
            elapsed = elapsed_previous + (time.perf_counter() - run_start)
            save_memory_checkpoint_atomic(
                memory_dnn,
                resume_path,
                {
                    "stage": "pretraining",
                    "n_users": int(n_users),
                    "completed_frames": int(completed_now),
                    "selected_ranks": selected_ranks,
                    "rewards": rewards[:completed_now],
                    "k_history": k_history[:completed_now],
                    "elapsed_seconds": float(elapsed),
                    "config": config,
                    "saved_at_utc": utc_now(),
                },
            )

        if completed_now % 1000 == 0 or completed_now == required_frames:
            recent_start = max(0, completed_now - 1000)
            print(
                f"  pre-training {completed_now:>6}/{required_frames}: "
                f"K={current_k}, recent mean reward="
                f"{np.mean(rewards[recent_start:completed_now]):.6e}"
            )

    elapsed_total = elapsed_previous + (time.perf_counter() - run_start)
    if memory_dnn.memory_counter != required_frames:
        raise AssertionError(
            f"expected {required_frames} encoded frames, got {memory_dnn.memory_counter}"
        )
    return {
        "memory_dnn": memory_dnn,
        "elapsed_seconds": float(elapsed_total),
        "rewards": rewards,
        "k_history": k_history,
        "selected_ranks": np.asarray(selected_ranks, dtype=np.int16),
        "mean_reward": float(np.mean(rewards)),
        "last_1000_mean_reward": float(np.mean(rewards[-1000:])),
        "final_k": int(k_history[-1]),
    }


def save_label_resume(
    path: Path,
    completed: int,
    network_hash: str,
    config: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    elapsed_seconds: float,
) -> None:
    save_npz_atomic(
        path,
        completed_frames=np.asarray([completed], dtype=np.int64),
        network_hash=np.asarray([network_hash]),
        config_json=np.asarray([json.dumps(config, sort_keys=True)]),
        elapsed_seconds=np.asarray([elapsed_seconds], dtype=np.float64),
        **{name: value[:completed] for name, value in arrays.items()},
    )


def generate_frozen_labels(
    memory_dnn: MemoryDNN,
    channels: np.ndarray,
    n_users: int,
    config: Dict[str, Any],
    resume_path: Path,
    resume: bool,
    checkpoint_interval: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not memory_dnn.frozen:
        raise RuntimeError("teacher must be frozen before label generation")
    if any(parameter.requires_grad for parameter in memory_dnn.model.parameters()):
        raise RuntimeError("frozen teacher still has trainable parameters")

    network_hash_before = model_parameter_sha256(memory_dnn)
    num_frames = channels.shape[0]
    arrays = {
        "output_mode": np.empty((num_frames, n_users), dtype=np.int8),
        "output_obj": np.empty(num_frames, dtype=np.float64),
        "output_a": np.empty(num_frames, dtype=np.float64),
        "output_tau": np.empty((num_frames, n_users), dtype=np.float64),
        "adaptive_K": np.empty(num_frames, dtype=np.int16),
        "selected_candidate_rank": np.empty(num_frames, dtype=np.int16),
        "candidate_count": np.empty(num_frames, dtype=np.int16),
        "decode_time_seconds": np.empty(num_frames, dtype=np.float64),
        "candidate_evaluation_time_seconds": np.empty(num_frames, dtype=np.float64),
        "end_to_end_time_seconds": np.empty(num_frames, dtype=np.float64),
    }

    completed = 0
    elapsed_previous = 0.0
    if resume and resume_path.is_file():
        with np.load(str(resume_path), allow_pickle=False) as saved:
            completed = int(saved["completed_frames"][0])
            if str(saved["network_hash"][0]) != network_hash_before:
                raise ValueError("label resume checkpoint uses another teacher network")
            if str(saved["config_json"][0]) != json.dumps(config, sort_keys=True):
                raise ValueError("label resume checkpoint configuration mismatch")
            elapsed_previous = float(saved["elapsed_seconds"][0])
            for name in arrays:
                arrays[name][:completed] = saved[name]
        print(f"  resuming frozen labeling from {completed}/{num_frames}")

    selected_ranks = [
        int(value) for value in arrays["selected_candidate_rank"][:completed]
    ]
    current_k = n_users if completed == 0 else int(arrays["adaptive_K"][completed - 1])

    # Untimed warm-up is excluded from saved per-frame timing.
    if completed == 0:
        warm_candidates = memory_dnn.decode(
            channels[0] * float(config["channel_scale"]), n_users, mode="OP"
        )
        evaluate_candidates(channels[0], warm_candidates, n_users)

    run_start = time.perf_counter()
    for frame_index in range(completed, num_frames):
        if frame_index > 0 and frame_index % int(config["adaptive_k_interval"]) == 0:
            current_k = update_adaptive_k(
                selected_ranks, n_users, int(config["adaptive_k_interval"])
            )

        channel = channels[frame_index]
        scaled_channel = channel * float(config["channel_scale"])
        frame_start = time.perf_counter()
        decode_start = time.perf_counter()
        candidates = memory_dnn.decode(scaled_channel, current_k, mode="OP")
        decode_elapsed = time.perf_counter() - decode_start

        evaluation_start = time.perf_counter()
        best_index, mode, reward, a_value, tau_full = evaluate_candidates(
            channel, candidates, n_users
        )
        evaluation_elapsed = time.perf_counter() - evaluation_start
        frame_elapsed = time.perf_counter() - frame_start

        arrays["output_mode"][frame_index] = mode
        arrays["output_obj"][frame_index] = reward
        arrays["output_a"][frame_index] = a_value
        arrays["output_tau"][frame_index] = tau_full
        arrays["adaptive_K"][frame_index] = current_k
        arrays["selected_candidate_rank"][frame_index] = best_index + 1
        arrays["candidate_count"][frame_index] = len(candidates)
        arrays["decode_time_seconds"][frame_index] = decode_elapsed
        arrays["candidate_evaluation_time_seconds"][frame_index] = evaluation_elapsed
        arrays["end_to_end_time_seconds"][frame_index] = frame_elapsed
        selected_ranks.append(best_index + 1)
        completed_now = frame_index + 1

        if (
            completed_now % int(checkpoint_interval) == 0
            or completed_now == num_frames
        ):
            elapsed = elapsed_previous + (time.perf_counter() - run_start)
            save_label_resume(
                resume_path,
                completed_now,
                network_hash_before,
                config,
                arrays,
                elapsed,
            )

        if completed_now % 1000 == 0 or completed_now == num_frames:
            print(
                f"  frozen labeling {completed_now:>6}/{num_frames}: "
                f"K={current_k}, mean reward="
                f"{np.mean(arrays['output_obj'][:completed_now]):.6e}"
            )

    elapsed_total = elapsed_previous + (time.perf_counter() - run_start)
    network_hash_after = model_parameter_sha256(memory_dnn)
    if network_hash_after != network_hash_before:
        raise AssertionError("teacher parameters changed during frozen labeling")
    if memory_dnn.memory_counter != int(config["pretrain_frames"]):
        raise AssertionError("replay-memory counter changed during frozen labeling")

    validate_label_arrays(channels, arrays, n_users)
    metadata = {
        "elapsed_seconds": float(elapsed_total),
        "network_hash_before": network_hash_before,
        "network_hash_after": network_hash_after,
        "network_unchanged": True,
        "mean_reward": float(np.mean(arrays["output_obj"])),
        "initial_k": int(n_users),
        "final_k": int(arrays["adaptive_K"][-1]),
        "minimum_k": int(np.min(arrays["adaptive_K"])),
        "maximum_k": int(np.max(arrays["adaptive_K"])),
        "mean_k": float(np.mean(arrays["adaptive_K"])),
        "average_decode_time_ms": float(
            1000.0 * np.mean(arrays["decode_time_seconds"])
        ),
        "average_candidate_evaluation_time_ms": float(
            1000.0 * np.mean(arrays["candidate_evaluation_time_seconds"])
        ),
        "average_end_to_end_time_ms": float(
            1000.0 * np.mean(arrays["end_to_end_time_seconds"])
        ),
    }
    return arrays, metadata


def validate_label_arrays(
    channels: np.ndarray,
    arrays: Dict[str, np.ndarray],
    n_users: int,
) -> None:
    num_frames = channels.shape[0]
    expected_shapes = {
        "output_mode": (num_frames, n_users),
        "output_obj": (num_frames,),
        "output_a": (num_frames,),
        "output_tau": (num_frames, n_users),
        "adaptive_K": (num_frames,),
        "selected_candidate_rank": (num_frames,),
        "candidate_count": (num_frames,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise AssertionError(f"{name} shape {arrays[name].shape}, expected {shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise AssertionError(f"{name} contains NaN or infinity")
    if not np.all(np.logical_or(arrays["output_mode"] == 0, arrays["output_mode"] == 1)):
        raise AssertionError("output_mode is not binary")
    if np.any(arrays["output_obj"] < 0.0):
        raise AssertionError("negative objective found")
    if np.any(arrays["output_a"] < -1e-10) or np.any(arrays["output_a"] > 1.0 + 1e-10):
        raise AssertionError("invalid energy-transfer fraction found")
    if np.any(arrays["output_tau"] < -1e-10):
        raise AssertionError("negative uplink allocation found")
    time_sum = arrays["output_a"] + np.sum(arrays["output_tau"], axis=1)
    if np.any(time_sum > 1.0 + 1e-7):
        raise AssertionError("time constraint violated in saved labels")
    if np.any(arrays["output_tau"][arrays["output_mode"] == 0] != 0.0):
        raise AssertionError("local UE received uplink time")
    if np.any(arrays["adaptive_K"] < 1) or np.any(arrays["adaptive_K"] > n_users):
        raise AssertionError("adaptive K outside [1,N]")
    if np.any(arrays["candidate_count"] != arrays["adaptive_K"]):
        raise AssertionError("candidate count differs from adaptive K")
    if np.any(arrays["selected_candidate_rank"] < 1) or np.any(
        arrays["selected_candidate_rank"] > arrays["candidate_count"]
    ):
        raise AssertionError("selected candidate rank is outside candidate set")

    # Deterministically re-solve a compact audit subset, including split edges.
    split = split_indices(num_frames)
    audit_indices = sorted(
        {
            0,
            max(0, split["fit_end"] - 1),
            split["fit_end"],
            max(0, split["development_end"] - 1),
            split["development_end"],
            num_frames - 1,
            *np.linspace(0, num_frames - 1, num=min(20, num_frames), dtype=int).tolist(),
        }
    )
    for index in audit_indices:
        gain, a_value, _, details = bisection(
            channels[index],
            arrays["output_mode"][index],
            return_details=True,
            delta=DEFAULT_BISECTION_DELTA,
        )
        if not np.isclose(gain, arrays["output_obj"][index], rtol=1e-11, atol=1e-5):
            raise AssertionError(f"objective audit failed at frame {index}")
        if not np.isclose(a_value, arrays["output_a"][index], rtol=1e-11, atol=1e-10):
            raise AssertionError(f"allocation-a audit failed at frame {index}")
        if not np.allclose(
            details["tau_full"], arrays["output_tau"][index], rtol=1e-11, atol=1e-10
        ):
            raise AssertionError(f"allocation-tau audit failed at frame {index}")


def publish_outputs(
    root: Path,
    n_users: int,
    channels: np.ndarray,
    arrays: Dict[str, np.ndarray],
    config: Dict[str, Any],
    data_path: Path,
    data_hash: str,
    protocol_path: Path,
    protocol_hash: str,
    frozen_checkpoint: Path,
    pretraining: Dict[str, Any],
    labeling: Dict[str, Any],
) -> Tuple[Path, Path]:
    output_dir = root / "DROO_labels"
    mat_path = output_dir / f"droo_teacher_labels_N{n_users}.mat"
    json_path = output_dir / f"droo_teacher_labels_N{n_users}_config.json"
    split = split_indices(channels.shape[0])
    split_role = np.empty(channels.shape[0], dtype=np.int8)
    split_role[: split["fit_end"]] = 0
    split_role[split["fit_end"] : split["development_end"]] = 1
    split_role[split["development_end"] :] = 2

    mat_payload: Dict[str, Any] = {
        "input_h": channels,
        "output_mode": arrays["output_mode"],
        "output_obj": arrays["output_obj"].reshape(-1, 1),
        "output_a": arrays["output_a"].reshape(-1, 1),
        "output_tau": arrays["output_tau"],
        "adaptive_K": arrays["adaptive_K"].reshape(-1, 1),
        "selected_candidate_rank": arrays["selected_candidate_rank"].reshape(-1, 1),
        "candidate_count": arrays["candidate_count"].reshape(-1, 1),
        "split_role": split_role.reshape(-1, 1),
        "fit_end_zero_based_exclusive": np.asarray([[split["fit_end"]]], dtype=np.int64),
        "development_end_zero_based_exclusive": np.asarray(
            [[split["development_end"]]], dtype=np.int64
        ),
        "teacher_parameter_sha256": labeling["network_hash_after"],
        "protocol_name": PROTOCOL_NAME,
        "generator_version": GENERATOR_VERSION,
        "configuration_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }
    atomic_savemat(mat_path, mat_payload)

    audit = {
        "status": "complete_and_validated",
        "generated_at_utc": utc_now(),
        "protocol_name": PROTOCOL_NAME,
        "generator_version": GENERATOR_VERSION,
        "teacher": "frozen adaptive-K reinforcement-learning teacher",
        "n_users": int(n_users),
        "num_frames": int(channels.shape[0]),
        "split": split,
        "configuration": config,
        "source_data": {
            "path": str(data_path),
            "sha256": data_hash,
        },
        "protocol_file": {
            "path": str(protocol_path),
            "sha256": protocol_hash,
        },
        "frozen_teacher_checkpoint": {
            "path": str(frozen_checkpoint),
            "sha256": sha256_file(frozen_checkpoint),
            "parameter_sha256": labeling["network_hash_after"],
        },
        "pretraining": {
            "elapsed_seconds": float(pretraining["elapsed_seconds"]),
            "mean_reward": float(pretraining["mean_reward"]),
            "last_1000_mean_reward": float(pretraining["last_1000_mean_reward"]),
            "final_k": int(pretraining["final_k"]),
            "encoded_frames": int(config["pretrain_frames"]),
        },
        "labeling": labeling,
        "validation": {
            "network_frozen": True,
            "network_unchanged_during_labeling": True,
            "no_encode_or_learn_during_labeling": True,
            "all_modes_binary": True,
            "all_allocations_feasible": True,
            "audit_subset_recomputed": True,
        },
        "environment": environment_metadata(),
        "mat_file": {
            "path": str(mat_path),
            "sha256": sha256_file(mat_path),
            "saved_variables": sorted(mat_payload.keys()),
        },
    }
    atomic_json(json_path, audit)
    return mat_path, json_path


def completed_output_is_valid(root: Path, n_users: int) -> bool:
    output_dir = root / "DROO_labels"
    mat_path = output_dir / f"droo_teacher_labels_N{n_users}.mat"
    json_path = output_dir / f"droo_teacher_labels_N{n_users}_config.json"
    data_path = root / "data" / f"data_{n_users}.mat"
    protocol_path = root / "configs" / "experiment_protocol.json"
    frozen_checkpoint = (
        output_dir / "checkpoints" / f"N{n_users}" / f"frozen_teacher_N{n_users}.pt"
    )
    required_paths = (
        mat_path,
        json_path,
        data_path,
        protocol_path,
        frozen_checkpoint,
    )
    if not all(path.is_file() for path in required_paths):
        return False
    try:
        audit = json.loads(json_path.read_text(encoding="utf-8"))
        return (
            audit.get("status") == "complete_and_validated"
            and audit.get("protocol_name") == PROTOCOL_NAME
            and int(audit.get("n_users", -1)) == n_users
            and audit.get("mat_file", {}).get("sha256") == sha256_file(mat_path)
            and audit.get("source_data", {}).get("sha256") == sha256_file(data_path)
            and audit.get("protocol_file", {}).get("sha256")
            == sha256_file(protocol_path)
            and audit.get("frozen_teacher_checkpoint", {}).get("sha256")
            == sha256_file(frozen_checkpoint)
            and bool(
                audit.get("validation", {}).get(
                    "network_unchanged_during_labeling", False
                )
            )
        )
    except Exception:
        return False


def run_for_n(
    root: Path,
    n_users: int,
    pretrain_frames: int,
    seed: int,
    resume: bool,
    force: bool,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    if completed_output_is_valid(root, n_users) and not force:
        print(f"\nN={n_users}: validated formal outputs already exist; skipping.")
        output_dir = root / "DROO_labels"
        mat_path = output_dir / f"droo_teacher_labels_N{n_users}.mat"
        json_path = output_dir / f"droo_teacher_labels_N{n_users}_config.json"
        frozen_checkpoint = (
            output_dir
            / "checkpoints"
            / f"N{n_users}"
            / f"frozen_teacher_N{n_users}.pt"
        )
        audit = json.loads(json_path.read_text(encoding="utf-8"))
        return {
            "n_users": n_users,
            "status": "existing_valid_output",
            "mat": str(mat_path),
            "json": str(json_path),
            "frozen_checkpoint": str(frozen_checkpoint),
            "mat_sha256": sha256_file(mat_path),
            "config_sha256": sha256_file(json_path),
            "checkpoint_sha256": sha256_file(frozen_checkpoint),
            "source_data_sha256": audit["source_data"]["sha256"],
            "protocol_sha256": audit["protocol_file"]["sha256"],
            "network_parameter_sha256": audit["frozen_teacher_checkpoint"][
                "parameter_sha256"
            ],
        }

    set_global_seed(seed)
    protocol_path = root / "configs" / "experiment_protocol.json"
    protocol_hash = sha256_file(protocol_path)
    data_path = root / "data" / f"data_{n_users}.mat"
    data_hash = sha256_file(data_path)
    channels = load_channels(data_path, n_users)
    split = split_indices(channels.shape[0])
    if int(pretrain_frames) > split["development_end"]:
        raise ValueError(
            f"N={n_users}: pretraining frames ({pretrain_frames}) would enter the "
            f"final test suffix beginning at {split['development_end']}"
        )

    config = make_config(n_users, pretrain_frames, seed)
    checkpoint_dir = root / "DROO_labels" / "checkpoints" / f"N{n_users}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrain_resume = checkpoint_dir / "pretraining_resume.pt"
    frozen_checkpoint = checkpoint_dir / f"frozen_teacher_N{n_users}.pt"
    label_resume = checkpoint_dir / "labeling_resume.npz"

    print(f"\nN={n_users}: L={channels.shape[0]}, development_end={split['development_end']}")
    if frozen_checkpoint.is_file() and resume and not force:
        payload = torch_load_complete(frozen_checkpoint)
        metadata = dict(payload.get("extra_metadata", {}))
        if metadata.get("config") != config:
            raise ValueError(f"frozen checkpoint configuration mismatch: {frozen_checkpoint}")
        memory_dnn = MemoryDNN.load_checkpoint(frozen_checkpoint, map_location="cpu")
        if not memory_dnn.frozen:
            raise ValueError(f"saved formal teacher is not frozen: {frozen_checkpoint}")
        pretraining = dict(metadata["pretraining_summary"])
        print(f"  loaded frozen teacher checkpoint: {frozen_checkpoint}")
    else:
        print("  pre-training MemoryDNN on the chronological development prefix...")
        memory_dnn = make_memory_dnn(n_users, config)
        pretraining = pretrain_teacher(
            memory_dnn,
            channels[:pretrain_frames],
            n_users,
            config,
            pretrain_resume,
            resume=resume and not force,
            checkpoint_interval=checkpoint_interval,
        )
        memory_dnn = pretraining.pop("memory_dnn")
        memory_dnn.freeze()
        frozen_summary = {
            "elapsed_seconds": float(pretraining["elapsed_seconds"]),
            "mean_reward": float(pretraining["mean_reward"]),
            "last_1000_mean_reward": float(pretraining["last_1000_mean_reward"]),
            "final_k": int(pretraining["final_k"]),
        }
        save_memory_checkpoint_atomic(
            memory_dnn,
            frozen_checkpoint,
            {
                "stage": "frozen_teacher",
                "n_users": int(n_users),
                "config": config,
                "source_data_sha256": data_hash,
                "protocol_sha256": protocol_hash,
                "parameter_sha256": model_parameter_sha256(memory_dnn),
                "pretraining_summary": frozen_summary,
                "saved_at_utc": utc_now(),
            },
        )
        pretraining = frozen_summary
        print(f"  frozen teacher saved: {frozen_checkpoint}")

    print("  generating labels from the frozen teacher (adaptive K reset to N)...")
    arrays, labeling = generate_frozen_labels(
        memory_dnn,
        channels,
        n_users,
        config,
        label_resume,
        resume=resume and not force,
        checkpoint_interval=checkpoint_interval,
    )
    mat_path, json_path = publish_outputs(
        root,
        n_users,
        channels,
        arrays,
        config,
        data_path,
        data_hash,
        protocol_path,
        protocol_hash,
        frozen_checkpoint,
        pretraining,
        labeling,
    )
    print(f"N={n_users}: COMPLETE -> {mat_path}")
    return {
        "n_users": int(n_users),
        "status": "generated_and_validated",
        "mat": str(mat_path),
        "json": str(json_path),
        "frozen_checkpoint": str(frozen_checkpoint),
        "mat_sha256": sha256_file(mat_path),
        "network_parameter_sha256": labeling["network_hash_after"],
    }


def run_self_test(root: Path) -> None:
    print("Teacher generator self-test (no formal output will be written)")
    channels = load_channels(root / "data" / "data_10.mat", 10)[:24]
    config = make_config(10, pretrain_frames=20, seed=42)
    config["batch_size"] = 8
    config["memory_size"] = 32
    config["training_interval"] = 2
    config["adaptive_k_interval"] = 4
    temporary_dir = root / "logs" / "teacher_generator_selftest"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    memory_dnn = make_memory_dnn(10, config)
    summary = pretrain_teacher(
        memory_dnn,
        channels,
        10,
        config,
        temporary_dir / "pretraining_resume.pt",
        resume=False,
        checkpoint_interval=10,
    )
    memory_dnn = summary["memory_dnn"]
    memory_dnn.freeze()
    before = model_parameter_sha256(memory_dnn)
    arrays, metadata = generate_frozen_labels(
        memory_dnn,
        channels[:8],
        10,
        config,
        temporary_dir / "labeling_resume.npz",
        resume=False,
        checkpoint_interval=4,
    )
    after = model_parameter_sha256(memory_dnn)
    if before != after or not metadata["network_unchanged"]:
        raise AssertionError("self-test teacher changed during labeling")
    if arrays["output_mode"].shape != (8, 10):
        raise AssertionError("self-test produced an invalid label shape")
    print("Teacher generator self-test: PASSED")
    print(f"  modes shape: {arrays['output_mode'].shape}")
    print(f"  frozen parameter hash: {after}")
    print(f"  self-test files: {temporary_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain, freeze, checkpoint, and run the adaptive-K RL teacher."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--n-users", type=int, nargs="+", default=list(DEFAULT_USERS)
    )
    parser.add_argument("--pretrain-frames", type=int, default=PRETRAIN_FRAMES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoints in this new experiment directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly regenerate a completed N (normally not allowed).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny non-formal integration test and exit.",
    )
    args = parser.parse_args()
    if any(int(value) <= 0 for value in args.n_users):
        parser.error("all --n-users values must be positive")
    if tuple(args.n_users) != DEFAULT_USERS:
        parser.error("this public artifact is intentionally limited to N=10")
    if int(args.pretrain_frames) <= 0:
        parser.error("--pretrain-frames must be positive")
    if int(args.checkpoint_interval) <= 0:
        parser.error("--checkpoint-interval must be positive")
    if args.force and not args.no_resume:
        parser.error("use --force together with --no-resume to regenerate cleanly")
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    load_protocol(root)
    if args.self_test:
        run_self_test(root)
        return

    output_dir = root / "DROO_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Frozen adaptive-K RL-teacher label generation")
    print(f"Protocol: {PROTOCOL_NAME}")
    print(f"Root: {root}")
    print(f"Output: {output_dir}")
    print(
        "Settings: N-120-80-N, pretrain=24000, lr=0.01, interval=10, "
        "batch=128, memory=1024, initial K=N, Delta_K=32, seed=42"
    )
    print("Formal final-test labels are generated only after network freezing.")

    records: List[Dict[str, Any]] = []
    for n_users in args.n_users:
        records.append(
            run_for_n(
                root=root,
                n_users=int(n_users),
                pretrain_frames=int(args.pretrain_frames),
                seed=int(args.seed),
                resume=not args.no_resume,
                force=bool(args.force),
                checkpoint_interval=int(args.checkpoint_interval),
            )
        )

    manifest = {
        "status": "complete",
        "protocol_name": PROTOCOL_NAME,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": utc_now(),
        "root": str(root),
        "n_users": [int(value) for value in args.n_users],
        "seed": int(args.seed),
        "pretrain_frames": int(args.pretrain_frames),
        "resume_enabled": not args.no_resume,
        "records": records,
        "environment": environment_metadata(),
    }
    manifest_path = output_dir / "droo_teacher_labels_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"\nAll requested N completed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
