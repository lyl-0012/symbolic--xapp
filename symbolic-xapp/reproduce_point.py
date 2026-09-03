"""Reproduce the released N=10 symbolic-controller test point.

This script fits only the proposed Full-feature symbolic classifier chain and
evaluates it on the chronologically indexed evaluation suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import scipy
import sklearn
from scipy.io import loadmat

from symbolic_xapp.resource_allocator import bisection
from symbolic_xapp.rule_model import RulePolicyModel


PROTOCOL_PATH = ROOT / "configs" / "experiment_protocol.json"
TEACHER_PATH = ROOT / "DROO_labels" / "droo_teacher_labels_N10.mat"
RESULTS_DIR = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    point = dict(protocol["released_point"])
    if protocol.get("release_scope") != "single-point-symbolic-controller":
        raise RuntimeError("unexpected release scope")
    if int(point["n_users"]) != 10 or int(point["rule_seed"]) != 42:
        raise RuntimeError("this artifact is locked to N=10 and seed=42")
    data_path = ROOT / protocol["data"]["path"]
    actual_hash = sha256_file(data_path)
    if actual_hash != protocol["data"]["sha256"]:
        raise RuntimeError(
            f"channel-data hash mismatch: {actual_hash} != "
            f"{protocol['data']['sha256']}"
        )
    return protocol, point


def load_teacher() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not TEACHER_PATH.is_file():
        raise FileNotFoundError(
            "Frozen teacher labels are missing. Run "
            "`python scripts/generate_teacher_labels.py --root .` first."
        )
    payload = loadmat(
        TEACHER_PATH,
        variable_names=[
            "input_h",
            "output_mode",
            "output_obj",
            "development_end_zero_based_exclusive",
        ],
    )
    channels = np.asarray(payload["input_h"], dtype=np.float64)
    modes = np.asarray(payload["output_mode"], dtype=np.int8)
    objectives = np.asarray(payload["output_obj"], dtype=np.float64).reshape(-1)
    development_end = int(
        np.asarray(payload["development_end_zero_based_exclusive"]).reshape(-1)[0]
    )
    if channels.shape != (30000, 10) or modes.shape != channels.shape:
        raise ValueError(
            f"unexpected teacher arrays: channels={channels.shape}, modes={modes.shape}"
        )
    if objectives.shape != (30000,) or development_end != 24000:
        raise ValueError("unexpected teacher objectives or chronological split")
    if not np.all(np.isin(modes, (0, 1))):
        raise ValueError("teacher modes are not binary")
    if np.any(objectives <= 0.0) or not np.all(np.isfinite(objectives)):
        raise ValueError("teacher objectives are invalid")
    return channels, modes, objectives, development_end


def evaluate_utility(channels: np.ndarray, modes: np.ndarray) -> np.ndarray:
    objectives = np.empty(len(channels), dtype=np.float64)
    for index, (channel, mode) in enumerate(zip(channels, modes), start=1):
        objectives[index - 1] = float(bisection(channel, mode, delta=0.005)[0])
        if index % 500 == 0 or index == len(channels):
            print(f"  allocation evaluation: {index}/{len(channels)}")
    return objectives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-test-frames",
        type=int,
        default=None,
        help="diagnostic only: evaluate the first K test frames instead of all 6000",
    )
    args = parser.parse_args()
    if args.quick_test_frames is not None and not 1 <= args.quick_test_frames <= 6000:
        parser.error("--quick-test-frames must be between 1 and 6000")
    return args


def main() -> None:
    args = parse_args()
    protocol, point = load_protocol()
    channels, teacher_modes, teacher_objectives, development_end = load_teacher()

    model = RulePolicyModel(
        n_users=10,
        max_depth=int(point["max_depth"]),
        min_samples_leaf=int(point["min_samples_leaf"]),
        random_state=int(point["rule_seed"]),
        chain_enabled=True,
        enabled_feature_groups=tuple(point["feature_groups"]),
        class_weight=None,
        chain_training="cross_fitted",
        oof_folds=int(point["oof_folds"]),
    )

    print("Fitting the released symbolic controller on frames 0:24000...")
    fit_start = time.perf_counter()
    model.fit(channels[:development_end], teacher_modes[:development_end])
    fit_seconds = time.perf_counter() - fit_start

    stop = len(channels)
    if args.quick_test_frames is not None:
        stop = development_end + int(args.quick_test_frames)
    test_channels = channels[development_end:stop]
    test_teacher_modes = teacher_modes[development_end:stop]
    test_teacher_objectives = teacher_objectives[development_end:stop]

    rule_modes = model.predict_from_rules(test_channels)
    fidelity = model.evaluate(test_channels, test_teacher_modes)
    print(f"Evaluating one conditional allocation per test frame ({len(test_channels)})...")
    rule_objectives = evaluate_utility(test_channels, rule_modes)

    teacher_sum = float(np.sum(test_teacher_objectives))
    utility_retention = float(np.sum(rule_objectives) / teacher_sum)
    total_rules = int(sum(len(item["rules"]) for item in model.rule_texts))
    formal = args.quick_test_frames is None
    metrics = {
        "status": "complete_and_validated",
        "formal_single_point": formal,
        "protocol_name": protocol["protocol_name"],
        "N": 10,
        "seed": 42,
        "features": point["feature_groups"],
        "max_depth": 7,
        "min_samples_leaf": 10,
        "oof_folds": 5,
        "num_development_frames": development_end,
        "num_test_frames": int(len(test_channels)),
        "utility_retention_ratio_of_means": utility_retention,
        "utility_retention_percent": 100.0 * utility_retention,
        "joint_action_fidelity": float(fidelity["joint_fidelity"]),
        "bitwise_fidelity": float(fidelity["bitwise_fidelity"]),
        "rule_tree_joint_fidelity": float(
            fidelity["tree_rule_serialization_fidelity"]["joint_fidelity"]
        ),
        "total_executable_rule_count": total_rules,
        "teacher_average_utility": float(np.mean(test_teacher_objectives)),
        "rule_average_utility": float(np.mean(rule_objectives)),
        "fit_seconds_diagnostic": float(fit_seconds),
        "inputs": {
            "teacher_labels": str(TEACHER_PATH.relative_to(ROOT)),
            "teacher_labels_sha256": sha256_file(TEACHER_PATH),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "thread_count": 1,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "formal" if formal else f"diagnostic_{len(test_channels)}"
    rules_path = RESULTS_DIR / f"rules_N10_seed42_{suffix}.json"
    metrics_path = RESULTS_DIR / f"metrics_N10_seed42_{suffix}.json"
    arrays_path = RESULTS_DIR / f"per_frame_N10_seed42_{suffix}.npz"
    model.save_rules_json(rules_path)
    write_json(metrics_path, metrics)
    np.savez_compressed(
        arrays_path,
        test_frame_index=np.arange(development_end, stop, dtype=np.int64),
        rule_modes=rule_modes,
        teacher_modes=test_teacher_modes,
        rule_objectives=rule_objectives,
        teacher_objectives=test_teacher_objectives,
    )

    print("\nReleased point completed")
    print(f"  utility retention: {100.0 * utility_retention:.6f}%")
    print(f"  joint-action fidelity: {100.0 * fidelity['joint_fidelity']:.6f}%")
    print(f"  bitwise fidelity: {100.0 * fidelity['bitwise_fidelity']:.6f}%")
    print(f"  executable rules: {total_rules}")
    print(f"  metrics: {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
