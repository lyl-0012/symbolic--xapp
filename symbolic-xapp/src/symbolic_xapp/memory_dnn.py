"""Deterministic PyTorch implementation of the DROO MemoryDNN.

This module provides the neural candidate generator used only by the offline
reinforcement-learning teacher.  It preserves the familiar ``remember``,
``encode``, ``learn`` and ``decode`` API of the original implementation while
adding reproducible sampling, explicit freezing, and complete checkpoints.

The deployed symbolic controller does not import or execute this network.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ArrayLike = Union[np.ndarray, Sequence[float]]
CHECKPOINT_FORMAT = "memory-dnn-v2-complete-state"


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for CPU-reproducible execution."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, RuntimeError):
            # CPU operations used here are deterministic on supported PyTorch
            # releases.  Older releases may not expose this global switch.
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class MemoryDNN:
    """Replay-memory neural candidate generator for DROO.

    Parameters
    ----------
    net:
        Four layer widths ``[input, hidden_1, hidden_2, output]``.  For the
        paper experiments this is ``[N, 120, 80, N]``.
    learning_rate, training_interval, batch_size, memory_size:
        Offline-teacher training parameters.
    seed:
        Seed used both for parameter initialization and replay sampling.
    device:
        ``"cpu"`` is required for the formal latency/reproducibility protocol.
    adam_betas, weight_decay:
        Optimizer parameters.  Adam is constructed once so its moment state is
        retained across training updates.
    """

    def __init__(
        self,
        net: Sequence[int],
        learning_rate: float = 0.01,
        training_interval: int = 10,
        batch_size: int = 128,
        memory_size: int = 1024,
        output_graph: bool = False,
        seed: int = 42,
        device: str = "cpu",
        deterministic: bool = True,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1.0e-4,
    ) -> None:
        del output_graph  # retained only for backward-compatible construction

        self.net = [int(value) for value in net]
        if len(self.net) != 4:
            raise ValueError(f"net must contain four widths, got {self.net}")
        if any(width <= 0 for width in self.net):
            raise ValueError(f"all network widths must be positive, got {self.net}")
        if self.net[0] != self.net[-1]:
            raise ValueError(
                "DROO requires equal input and output widths (one per UE), "
                f"got {self.net[0]} and {self.net[-1]}"
            )

        self.lr = float(learning_rate)
        self.training_interval = int(training_interval)
        self.batch_size = int(batch_size)
        self.memory_size = int(memory_size)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.adam_betas = (float(adam_betas[0]), float(adam_betas[1]))
        self.weight_decay = float(weight_decay)

        if self.lr <= 0:
            raise ValueError("learning_rate must be positive")
        if self.training_interval <= 0:
            raise ValueError("training_interval must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if not (0.0 <= self.adam_betas[0] < 1.0 and 0.0 <= self.adam_betas[1] < 1.0):
            raise ValueError(f"invalid Adam betas: {self.adam_betas}")

        self.device = torch.device(device)
        if self.device.type != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device is unavailable: {self.device}")

        set_deterministic_seed(self.seed, self.deterministic)
        self._rng = np.random.RandomState(self.seed)

        self.enumerate_actions: Optional[np.ndarray] = None
        self.memory = np.zeros(
            (self.memory_size, self.net[0] + self.net[-1]), dtype=np.float32
        )
        self.memory_counter = 0       # number of entries ever written
        self.valid_memory_size = 0    # number of initialized replay rows
        self.training_steps = 0
        self.cost_his: List[float] = []
        self.cost: Optional[float] = None
        self.frozen = False

        self._build_net()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.adam_betas,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.BCELoss()

    def _build_net(self) -> None:
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.ReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.ReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid(),
        ).to(self.device)

    def _validate_channel(self, h: ArrayLike) -> np.ndarray:
        array = np.asarray(h, dtype=np.float32).reshape(-1)
        if array.shape != (self.net[0],):
            raise ValueError(
                f"channel vector must have shape ({self.net[0]},), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("channel vector contains NaN or infinity")
        return array

    def _validate_mode(self, m: ArrayLike) -> np.ndarray:
        array = np.asarray(m, dtype=np.float32).reshape(-1)
        if array.shape != (self.net[-1],):
            raise ValueError(
                f"mode vector must have shape ({self.net[-1]},), got {array.shape}"
            )
        if not np.all(np.logical_or(array == 0.0, array == 1.0)):
            raise ValueError("mode vector must be binary")
        return array

    def remember(self, h: ArrayLike, m: ArrayLike) -> None:
        """Insert one initialized transition into the cyclic replay memory."""

        if self.frozen:
            raise RuntimeError("cannot update replay memory after MemoryDNN.freeze()")
        channel = self._validate_channel(h)
        mode = self._validate_mode(m)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.concatenate((channel, mode))
        self.memory_counter += 1
        self.valid_memory_size = min(self.memory_counter, self.memory_size)

    def encode(self, h: ArrayLike, m: ArrayLike) -> Optional[float]:
        """Store one selected action and train at the configured interval."""

        if self.frozen:
            raise RuntimeError("encode is forbidden after MemoryDNN.freeze()")
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            return self.learn()
        return None

    def learn(self) -> float:
        """Perform one Adam update using only initialized replay entries."""

        if self.frozen:
            raise RuntimeError("learn is forbidden after MemoryDNN.freeze()")
        if self.valid_memory_size <= 0:
            raise RuntimeError("cannot learn from an empty replay memory")

        sample_index = self._rng.choice(
            self.valid_memory_size,
            size=self.batch_size,
            replace=True,
        )
        batch_memory = self.memory[sample_index, :]
        h_train = torch.as_tensor(
            batch_memory[:, : self.net[0]], dtype=torch.float32, device=self.device
        )
        m_train = torch.as_tensor(
            batch_memory[:, self.net[0] :], dtype=torch.float32, device=self.device
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(h_train)
        loss = self.criterion(prediction, m_train)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite MemoryDNN loss: {loss.item()}")
        loss.backward()
        self.optimizer.step()

        self.cost = float(loss.detach().cpu().item())
        self.cost_his.append(self.cost)
        self.training_steps += 1
        return self.cost

    def freeze(self) -> None:
        """Freeze the pretrained network before full-dataset label generation."""

        self.frozen = True
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        """Restore trainability; intended only for explicit checkpoint recovery."""

        self.frozen = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.model.train()

    def decode(
        self,
        h: ArrayLike,
        k: int = 1,
        mode: str = "OP",
        return_metadata: bool = False,
    ) -> Any:
        """Generate binary mode candidates without changing network state."""

        channel = self._validate_channel(h)
        requested_k = int(k)
        if requested_k <= 0:
            raise ValueError(f"k must be positive, got {requested_k}")

        mode = str(mode).upper()
        if mode == "OP" and requested_k > self.net[-1]:
            raise ValueError(
                f"OP supports at most N={self.net[-1]} candidates, got k={requested_k}"
            )
        if mode not in {"OP", "KNN"}:
            raise ValueError("mode must be 'OP' or 'KNN'")

        tensor = torch.as_tensor(
            channel.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        self.model.eval()
        with torch.no_grad():
            scores = self.model(tensor).detach().cpu().numpy()[0]

        if mode == "OP":
            candidates, metadata = self.knm(scores, requested_k, return_metadata=True)
        else:
            candidates, metadata = self.knn(scores, requested_k, return_metadata=True)

        if return_metadata:
            metadata.update(
                {
                    "decoded_scores": np.asarray(scores, dtype=float),
                    "threshold_distance": np.asarray(np.abs(scores - 0.5), dtype=float),
                    "requested_k": requested_k,
                    "returned_k": int(len(candidates)),
                    "mode": mode,
                    "frozen": bool(self.frozen),
                }
            )
            return candidates, metadata
        return candidates

    def knm(
        self, m: ArrayLike, k: int = 1, return_metadata: bool = False
    ) -> Any:
        """Generate the order-preserving candidates in DROO Eqs. (8)-(9)."""

        scores = np.asarray(m, dtype=float).reshape(-1)
        if scores.shape != (self.net[-1],):
            raise ValueError(f"score vector has invalid shape {scores.shape}")
        k = int(k)
        if not 1 <= k <= self.net[-1]:
            raise ValueError(f"OP requires 1 <= k <= {self.net[-1]}, got {k}")

        candidates: List[np.ndarray] = [(scores > 0.5).astype(np.int8)]
        sources: List[Dict[str, Any]] = [{"kind": "threshold"}]

        # Stable sorting makes equal-distance tie handling reproducible.
        nearest = np.argsort(np.abs(scores - 0.5), kind="mergesort")[: k - 1]
        for index in nearest:
            if scores[index] > 0.5:
                candidate = (scores - scores[index] > 0.0).astype(np.int8)
                direction = "to_zero"
            else:
                candidate = (scores - scores[index] >= 0.0).astype(np.int8)
                direction = "to_one"
            candidates.append(candidate)
            sources.append(
                {
                    "kind": "order_preserving_boundary",
                    "boundary_index": int(index),
                    "boundary_direction": direction,
                }
            )

        if return_metadata:
            return candidates, {"candidate_sources": sources}
        return candidates

    def knn(
        self, m: ArrayLike, k: int = 1, return_metadata: bool = False
    ) -> Any:
        """Return the nearest binary actions; practical only for small N."""

        scores = np.asarray(m, dtype=float).reshape(-1)
        if scores.shape != (self.net[-1],):
            raise ValueError(f"score vector has invalid shape {scores.shape}")
        k = int(k)
        total_actions = 2 ** self.net[-1]
        if not 1 <= k <= total_actions:
            raise ValueError(f"KNN requires 1 <= k <= {total_actions}, got {k}")
        if self.net[-1] > 20:
            raise ValueError("KNN enumeration is disabled for N>20; use OP mode")

        if self.enumerate_actions is None:
            import itertools

            self.enumerate_actions = np.asarray(
                list(itertools.product((0, 1), repeat=self.net[-1])), dtype=np.int8
            )
        squared_distance = np.square(self.enumerate_actions - scores).sum(axis=1)
        indices = np.argsort(squared_distance, kind="mergesort")[:k]
        candidates = self.enumerate_actions[indices].copy()

        if return_metadata:
            sources = [
                {
                    "kind": "nearest_neighbor",
                    "action_index": int(index),
                    "distance": float(squared_distance[index]),
                }
                for index in indices
            ]
            return candidates, {"candidate_sources": sources}
        return candidates

    def state_summary(self) -> Dict[str, Any]:
        """Return JSON-serializable state used by experiment manifests."""

        return {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "net": list(self.net),
            "learning_rate": self.lr,
            "training_interval": self.training_interval,
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "adam_betas": list(self.adam_betas),
            "weight_decay": self.weight_decay,
            "device": str(self.device),
            "memory_counter": int(self.memory_counter),
            "valid_memory_size": int(self.valid_memory_size),
            "training_steps": int(self.training_steps),
            "num_recorded_losses": int(len(self.cost_his)),
            "last_loss": None if self.cost is None else float(self.cost),
            "frozen": bool(self.frozen),
        }

    def save_checkpoint(
        self, path: Union[str, Path], extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save model, optimizer, replay memory and random-number state."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "configuration": {
                "net": list(self.net),
                "learning_rate": self.lr,
                "training_interval": self.training_interval,
                "batch_size": self.batch_size,
                "memory_size": self.memory_size,
                "seed": self.seed,
                "device": str(self.device),
                "deterministic": self.deterministic,
                "adam_betas": tuple(self.adam_betas),
                "weight_decay": self.weight_decay,
            },
            "model_state_dict": copy.deepcopy(self.model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "memory": self.memory.copy(),
            "memory_counter": int(self.memory_counter),
            "valid_memory_size": int(self.valid_memory_size),
            "training_steps": int(self.training_steps),
            "cost_his": list(self.cost_his),
            "cost": self.cost,
            "frozen": bool(self.frozen),
            "numpy_replay_rng_state": self._rng.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "extra_metadata": dict(extra_metadata or {}),
        }
        torch.save(payload, str(destination))
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: str = "cpu",
        restore_rng_state: bool = True,
    ) -> "MemoryDNN":
        """Reconstruct a complete MemoryDNN state from ``save_checkpoint``."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"MemoryDNN checkpoint not found: {source}")
        try:
            # PyTorch >=2.6 defaults to weights_only=True, whereas this complete
            # checkpoint intentionally also contains replay/RNG metadata.
            payload = torch.load(
                str(source), map_location=map_location, weights_only=False
            )
        except TypeError:
            # ``weights_only`` is unavailable in the user's PyTorch 2.0 build.
            payload = torch.load(str(source), map_location=map_location)
        if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"unsupported checkpoint format: {payload.get('checkpoint_format')!r}"
            )

        configuration = dict(payload["configuration"])
        configuration["device"] = map_location
        instance = cls(**configuration)
        instance.model.load_state_dict(payload["model_state_dict"], strict=True)
        instance.optimizer.load_state_dict(payload["optimizer_state_dict"])
        instance.memory[...] = np.asarray(payload["memory"], dtype=np.float32)
        instance.memory_counter = int(payload["memory_counter"])
        instance.valid_memory_size = int(payload["valid_memory_size"])
        instance.training_steps = int(payload["training_steps"])
        instance.cost_his = [float(value) for value in payload["cost_his"]]
        instance.cost = None if payload["cost"] is None else float(payload["cost"])
        instance._rng.set_state(payload["numpy_replay_rng_state"])

        if restore_rng_state:
            torch.set_rng_state(payload["torch_rng_state"].cpu())

        if bool(payload["frozen"]):
            instance.freeze()
        else:
            instance.unfreeze()
        return instance

__all__ = ["MemoryDNN", "set_deterministic_seed", "CHECKPOINT_FORMAT"]
