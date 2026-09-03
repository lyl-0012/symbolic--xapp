"""Executable symbolic policy for frozen RL-teacher distillation.

The deployed controller is a bounded-depth decision-tree classifier chain.
Training uses cross-fitted preceding predictions rather than true preceding
labels, reducing the train/inference mismatch of teacher-forced chains.  The
module exports both human-readable IF--THEN paths and a version-independent
JSON controller used for formal online execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier


MODEL_FORMAT = "cross-fitted-symbolic-chain-v1"
RULE_EXECUTION_NUMERIC_SEMANTICS = "sklearn-float32-threshold-v1"
CHANNEL_SCALE = 1.0e6
FEATURE_GROUPS = ("absolute", "relative", "rank", "global_stats")
GLOBAL_FEATURE_NAMES = (
    "h_mean",
    "h_std",
    "h_min",
    "h_max",
    "h_range",
)


def get_available_feature_groups() -> Tuple[str, ...]:
    return FEATURE_GROUPS


def environment_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def feature_group_from_name(feature_name: str) -> str:
    if feature_name.startswith("h_scaled_"):
        return "absolute"
    if feature_name.startswith("h_ratio_") or feature_name.startswith("h_centered_"):
        return "relative"
    if feature_name.startswith("h_rank_desc_"):
        return "rank"
    if feature_name in GLOBAL_FEATURE_NAMES:
        return "global_stats"
    if feature_name.startswith("prev_pred_UE"):
        return "chain"
    raise ValueError(f"unsupported feature name: {feature_name}")


def select_feature_indices(
    feature_names: Sequence[str],
    enabled_groups: Optional[Sequence[str]] = None,
) -> np.ndarray:
    groups = FEATURE_GROUPS if enabled_groups is None else tuple(enabled_groups)
    unknown = sorted(set(groups) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"unsupported feature groups: {unknown}")
    if len(set(groups)) != len(groups):
        raise ValueError(f"duplicate feature groups: {groups}")
    selected = [
        index
        for index, name in enumerate(feature_names)
        if feature_group_from_name(name) in groups
    ]
    if not selected:
        raise ValueError("no feature columns were selected")
    return np.asarray(selected, dtype=np.int64)


def _validate_channels(channels: np.ndarray, n_users: Optional[int] = None) -> np.ndarray:
    matrix = np.asarray(channels, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"channels must be a nonempty 2D matrix, got {matrix.shape}")
    if n_users is not None and matrix.shape[1] != int(n_users):
        raise ValueError(
            f"channels contain {matrix.shape[1]} users, expected {int(n_users)}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("channels contain NaN or infinity")
    if np.any(matrix <= 0.0):
        raise ValueError("channel power gains must be strictly positive")
    return np.ascontiguousarray(matrix)


def _average_descending_ranks(scaled_channels: np.ndarray) -> np.ndarray:
    """Rank 1 is strongest; exact ties receive the same average rank."""

    ranks = rankdata(-scaled_channels, method="average", axis=1)
    return np.asarray(ranks, dtype=np.float64)


def build_channel_feature_matrix(
    channels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Construct nonredundant absolute, relative, rank, and global features."""

    matrix = _validate_channels(channels)
    n_samples, n_users = matrix.shape
    scaled = np.maximum(matrix * CHANNEL_SCALE, 1.0e-12)
    mean = np.mean(scaled, axis=1, keepdims=True)
    total = np.sum(scaled, axis=1, keepdims=True)
    ratio = scaled / np.maximum(total, 1.0e-12)
    centered = scaled - mean
    ranks = _average_descending_ranks(scaled)
    minimum = np.min(scaled, axis=1)
    maximum = np.max(scaled, axis=1)
    global_stats = np.column_stack(
        (
            mean[:, 0],
            np.std(scaled, axis=1),
            minimum,
            maximum,
            maximum - minimum,
        )
    )

    features = np.hstack((scaled, ratio, centered, ranks, global_stats))
    names: List[str] = []
    for prefix in ("h_scaled", "h_ratio", "h_centered", "h_rank_desc"):
        names.extend(f"{prefix}_{user + 1}" for user in range(n_users))
    names.extend(GLOBAL_FEATURE_NAMES)
    if features.shape != (n_samples, len(names)):
        raise AssertionError(
            f"feature shape {features.shape} does not match {len(names)} names"
        )
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("constructed feature matrix is non-finite")
    return np.ascontiguousarray(features, dtype=np.float64), names


def validate_chain_order(
    n_users: int, chain_order: Optional[Sequence[int]]
) -> Tuple[int, ...]:
    if chain_order is None:
        return tuple(range(int(n_users)))
    order = tuple(int(value) for value in chain_order)
    if len(order) != int(n_users) or set(order) != set(range(int(n_users))):
        raise ValueError(
            f"chain_order must be a permutation of 0..{int(n_users)-1}, got {order}"
        )
    return order


def named_chain_order(n_users: int, name: str, seed: int = 42) -> Tuple[int, ...]:
    name = str(name).lower()
    if name == "forward":
        return tuple(range(int(n_users)))
    if name == "reverse":
        return tuple(reversed(range(int(n_users))))
    if name == "random":
        rng = np.random.RandomState(int(seed))
        return tuple(int(value) for value in rng.permutation(int(n_users)))
    raise ValueError("chain-order name must be forward, reverse, or random")


def _validate_modes(modes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    labels = np.asarray(modes)
    if labels.shape != shape:
        raise ValueError(f"mode shape {labels.shape} does not match {shape}")
    if not np.all(np.logical_or(labels == 0, labels == 1)):
        raise ValueError("modes must be binary")
    return np.ascontiguousarray(labels, dtype=np.int8)


def _compile_tree(
    classifier: DecisionTreeClassifier,
    feature_names: Sequence[str],
    user_index: int,
    chain_position: int,
) -> Dict[str, Any]:
    tree = classifier.tree_
    predicted_class = np.full(tree.node_count, -1, dtype=np.int16)
    leaf_mask = tree.children_left == tree.children_right
    for node in np.flatnonzero(leaf_mask):
        class_position = int(np.argmax(tree.value[node][0]))
        predicted_class[node] = int(classifier.classes_[class_position])
    return {
        "user_index": int(user_index),
        "chain_position": int(chain_position),
        "feature_names": list(feature_names),
        "children_left": tree.children_left.astype(np.int64).tolist(),
        "children_right": tree.children_right.astype(np.int64).tolist(),
        "feature": tree.feature.astype(np.int64).tolist(),
        "threshold": tree.threshold.astype(np.float64).tolist(),
        "predicted_class": predicted_class.astype(int).tolist(),
        "n_node_samples": tree.n_node_samples.astype(np.int64).tolist(),
        "weighted_n_node_samples": tree.weighted_n_node_samples.astype(np.float64).tolist(),
        "impurity": tree.impurity.astype(np.float64).tolist(),
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
    }


def _tree_leaf_rules(compiled_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    left = compiled_tree["children_left"]
    right = compiled_tree["children_right"]
    features = compiled_tree["feature"]
    thresholds = compiled_tree["threshold"]
    names = compiled_tree["feature_names"]
    predicted = compiled_tree["predicted_class"]
    samples = compiled_tree["n_node_samples"]
    rules: List[Dict[str, Any]] = []

    def visit(node: int, conditions: List[Dict[str, Any]]) -> None:
        if int(left[node]) == int(right[node]):
            rules.append(
                {
                    "conditions": list(conditions),
                    "predicted_class": int(predicted[node]),
                    "sample_count": int(samples[node]),
                    "leaf_node": int(node),
                }
            )
            return
        feature_index = int(features[node])
        condition = {
            "node_id": int(node),
            "feature_index": feature_index,
            "feature_name": names[feature_index],
            "threshold": float(thresholds[node]),
        }
        visit(
            int(left[node]),
            conditions + [{**condition, "operator": "<="}],
        )
        visit(
            int(right[node]),
            conditions + [{**condition, "operator": ">"}],
        )

    visit(0, [])
    return rules


def _predict_compiled_tree(
    compiled_tree: Dict[str, Any], feature_matrix: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=np.float32)
    left = np.asarray(compiled_tree["children_left"], dtype=np.int64)
    right = np.asarray(compiled_tree["children_right"], dtype=np.int64)
    feature = np.asarray(compiled_tree["feature"], dtype=np.int64)
    threshold = np.asarray(compiled_tree["threshold"], dtype=np.float64)
    predicted_class = np.asarray(compiled_tree["predicted_class"], dtype=np.int16)
    output = np.empty(matrix.shape[0], dtype=np.int8)
    for row_index, row in enumerate(matrix):
        node = 0
        while left[node] != right[node]:
            feature_index = feature[node]
            node = left[node] if row[feature_index] <= threshold[node] else right[node]
        prediction = int(predicted_class[node])
        if prediction not in (0, 1):
            raise RuntimeError(f"compiled leaf {node} has invalid prediction {prediction}")
        output[row_index] = prediction
    return output


def _fidelity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("fidelity inputs must be equally shaped 2D matrices")
    return {
        "num_samples": int(y_true.shape[0]),
        "bitwise_fidelity": float(np.mean(y_true == y_pred)),
        "joint_fidelity": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "per_user_fidelity": np.mean(y_true == y_pred, axis=0).astype(float).tolist(),
    }


@dataclass
class FitConfiguration:
    max_depth: Optional[int]
    min_samples_leaf: int
    random_state: int
    chain_enabled: bool
    chain_training: str
    oof_folds: int
    class_weight: Optional[str]


class RulePolicyModel:
    """Cross-fitted decision-tree classifier chain with executable rules."""

    def __init__(
        self,
        n_users: int,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        chain_enabled: bool = True,
        chain_order: Optional[Sequence[int]] = None,
        enabled_feature_groups: Optional[Sequence[str]] = None,
        class_weight: Optional[str] = None,
        chain_training: str = "cross_fitted",
        oof_folds: int = 5,
    ) -> None:
        self.n_users = int(n_users)
        if self.n_users <= 0:
            raise ValueError("n_users must be positive")
        self.max_depth = None if max_depth is None else int(max_depth)
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive or None")
        self.min_samples_leaf = int(min_samples_leaf)
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        self.random_state = int(random_state)
        self.chain_enabled = bool(chain_enabled)
        self.chain_order = validate_chain_order(self.n_users, chain_order)
        self.enabled_feature_groups = (
            FEATURE_GROUPS
            if enabled_feature_groups is None
            else tuple(enabled_feature_groups)
        )
        unknown = set(self.enabled_feature_groups) - set(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unsupported feature groups: {sorted(unknown)}")
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        self.class_weight = class_weight
        if chain_training not in ("cross_fitted", "teacher_forcing"):
            raise ValueError("chain_training must be cross_fitted or teacher_forcing")
        self.chain_training = chain_training
        self.oof_folds = int(oof_folds)
        if self.oof_folds < 2:
            raise ValueError("oof_folds must be at least two")

        self.models: List[DecisionTreeClassifier] = []
        self.compiled_trees: List[Dict[str, Any]] = []
        self.rule_texts: List[Dict[str, Any]] = []
        self.all_feature_names: List[str] = []
        self.feature_names: List[str] = []
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.fit_metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.is_fitted = False

    def _new_classifier(self, user_index: int, fold_offset: int = 0) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state + int(user_index) + 1009 * int(fold_offset),
        )

    def _cross_fitted_prediction(
        self,
        x_train: np.ndarray,
        target: np.ndarray,
        user_index: int,
    ) -> np.ndarray:
        if np.unique(target).size == 1:
            return np.full(target.shape, int(target[0]), dtype=np.int8)
        n_splits = min(self.oof_folds, len(target))
        if n_splits < 2:
            raise ValueError("not enough fitting samples for cross-fitting")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state + int(user_index),
        )
        prediction = np.empty(target.shape[0], dtype=np.int8)
        filled = np.zeros(target.shape[0], dtype=bool)
        for fold_index, (fit_indices, held_indices) in enumerate(splitter.split(x_train), 1):
            fold_model = self._new_classifier(user_index, fold_offset=fold_index)
            fold_model.fit(x_train[fit_indices], target[fit_indices])
            prediction[held_indices] = fold_model.predict(x_train[held_indices]).astype(np.int8)
            filled[held_indices] = True
        if not np.all(filled):
            raise AssertionError("cross-fitting did not predict every fitting sample")
        return prediction

    def fit(self, channels: np.ndarray, modes: np.ndarray) -> Dict[str, Any]:
        """Fit only on the explicitly supplied fitting partition."""

        channel_matrix = _validate_channels(channels, self.n_users)
        labels = _validate_modes(modes, channel_matrix.shape)
        if channel_matrix.shape[0] < max(10, self.oof_folds):
            raise ValueError("too few fitting samples")

        full_features, all_names = build_channel_feature_matrix(channel_matrix)
        selected = select_feature_indices(all_names, self.enabled_feature_groups)
        base_features = full_features[:, selected]
        self.all_feature_names = list(all_names)
        self.selected_feature_indices = selected
        self.feature_names = [all_names[index] for index in selected]

        oof_predictions = np.zeros_like(labels, dtype=np.int8)
        self.models = []
        self.compiled_trees = []
        self.rule_texts = []

        for chain_position, user_index in enumerate(self.chain_order):
            preceding_users = self.chain_order[:chain_position]
            if not self.chain_enabled or chain_position == 0:
                x_chain = base_features
                chain_names = list(self.feature_names)
            else:
                if self.chain_training == "cross_fitted":
                    previous = oof_predictions[:, preceding_users]
                else:
                    previous = labels[:, preceding_users]
                x_chain = np.hstack((base_features, previous))
                chain_names = list(self.feature_names) + [
                    f"prev_pred_UE{previous_user + 1}" for previous_user in preceding_users
                ]

            classifier = self._new_classifier(user_index)
            classifier.fit(x_chain, labels[:, user_index])
            self.models.append(classifier)
            if self.chain_training == "cross_fitted" and self.chain_enabled:
                oof_predictions[:, user_index] = self._cross_fitted_prediction(
                    x_chain, labels[:, user_index], user_index
                )
            else:
                oof_predictions[:, user_index] = classifier.predict(x_chain).astype(np.int8)

            compiled = _compile_tree(
                classifier, chain_names, user_index, chain_position
            )
            self.compiled_trees.append(compiled)
            self.rule_texts.append(
                {
                    "user_index": int(user_index),
                    "chain_position": int(chain_position),
                    "preceding_users": [int(value) for value in preceding_users],
                    "rules": _tree_leaf_rules(compiled),
                }
            )

        self.is_fitted = True
        deployed_fit_prediction = self.predict_from_rules(channel_matrix)
        sklearn_fit_prediction = self.predict(channel_matrix)
        serialization_fidelity = _fidelity(
            sklearn_fit_prediction, deployed_fit_prediction
        )
        if serialization_fidelity["joint_fidelity"] != 1.0:
            raise RuntimeError("exported symbolic controller differs from source trees")

        policy_fidelity = _fidelity(labels, deployed_fit_prediction)
        rule_lengths = [
            len(rule["conditions"])
            for user_rules in self.rule_texts
            for rule in user_rules["rules"]
        ]
        rule_counts = [len(item["rules"]) for item in self.rule_texts]
        self.fit_metadata = {
            "format": MODEL_FORMAT,
            "num_fit_samples": int(channel_matrix.shape[0]),
            "channels_sha256": array_sha256(channel_matrix),
            "modes_sha256": array_sha256(labels),
            "n_users": self.n_users,
            "feature_groups": list(self.enabled_feature_groups),
            "selected_feature_count": int(len(self.feature_names)),
            "chain_enabled": self.chain_enabled,
            "chain_order_zero_based": list(self.chain_order),
            "chain_order_one_based": [value + 1 for value in self.chain_order],
            "chain_training": self.chain_training,
            "oof_folds": self.oof_folds,
            "criterion": "gini",
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "numeric_semantics": RULE_EXECUTION_NUMERIC_SEMANTICS,
            "environment": environment_metadata(),
        }
        self.metrics = {
            "fit_policy_fidelity": policy_fidelity,
            "fit_tree_rule_serialization_fidelity": serialization_fidelity,
            "total_rule_count": int(sum(rule_counts)),
            "average_rules_per_user": float(np.mean(rule_counts)),
            "average_rule_length": float(np.mean(rule_lengths)) if rule_lengths else 0.0,
            "maximum_rule_length": int(max(rule_lengths)) if rule_lengths else 0,
        }
        return {"fit_metadata": self.fit_metadata, "metrics": self.metrics}

    def _prepare_features(self, channels: np.ndarray) -> Tuple[np.ndarray, bool]:
        raw = np.asarray(channels)
        single = raw.ndim == 1
        matrix = _validate_channels(raw, self.n_users)
        if self.selected_feature_indices is None:
            raise RuntimeError("model is not fitted")
        full, names = build_channel_feature_matrix(matrix)
        if names != self.all_feature_names:
            raise RuntimeError("feature schema changed after fitting")
        return full[:, self.selected_feature_indices], single

    def predict(self, channels: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(self.models) != self.n_users:
            raise RuntimeError("source tree model is not fitted/available")
        base, single = self._prepare_features(channels)
        output = np.zeros((base.shape[0], self.n_users), dtype=np.int8)
        for chain_position, (user_index, classifier) in enumerate(
            zip(self.chain_order, self.models)
        ):
            preceding = self.chain_order[:chain_position]
            x_chain = (
                base
                if not self.chain_enabled or chain_position == 0
                else np.hstack((base, output[:, preceding]))
            )
            output[:, user_index] = classifier.predict(x_chain).astype(np.int8)
        return output[0] if single else output

    def predict_from_rules(self, channels: np.ndarray) -> np.ndarray:
        if not self.is_fitted or len(self.compiled_trees) != self.n_users:
            raise RuntimeError("compiled symbolic controller is not available")
        base, single = self._prepare_features(channels)
        output = np.zeros((base.shape[0], self.n_users), dtype=np.int8)
        for chain_position, (user_index, compiled) in enumerate(
            zip(self.chain_order, self.compiled_trees)
        ):
            preceding = self.chain_order[:chain_position]
            x_chain = (
                base
                if not self.chain_enabled or chain_position == 0
                else np.hstack((base, output[:, preceding]))
            )
            output[:, user_index] = _predict_compiled_tree(compiled, x_chain)
        return output[0] if single else output

    def evaluate(self, channels: np.ndarray, modes: np.ndarray) -> Dict[str, Any]:
        matrix = _validate_channels(channels, self.n_users)
        labels = _validate_modes(modes, matrix.shape)
        rule_prediction = self.predict_from_rules(matrix)
        result = _fidelity(labels, rule_prediction)
        if self.models:
            source_prediction = self.predict(matrix)
            serialization = _fidelity(source_prediction, rule_prediction)
            if serialization["joint_fidelity"] != 1.0:
                raise RuntimeError("symbolic execution differs from source trees")
            result["tree_rule_serialization_fidelity"] = serialization
        return result

    def trace_paths(self, channels: np.ndarray) -> List[List[Dict[str, Any]]]:
        """Return executed conditions and path impurity reductions per sample/UE."""

        base, _ = self._prepare_features(channels)
        output = np.zeros((base.shape[0], self.n_users), dtype=np.int8)
        all_traces: List[List[Dict[str, Any]]] = []
        for row_index in range(base.shape[0]):
            sample_traces: List[Dict[str, Any]] = []
            for chain_position, (user_index, tree) in enumerate(
                zip(self.chain_order, self.compiled_trees)
            ):
                preceding = self.chain_order[:chain_position]
                values = (
                    base[row_index]
                    if not self.chain_enabled or chain_position == 0
                    else np.concatenate((base[row_index], output[row_index, preceding]))
                ).astype(np.float32, copy=False)
                left = tree["children_left"]
                right = tree["children_right"]
                feature = tree["feature"]
                threshold = tree["threshold"]
                impurity = tree["impurity"]
                weighted = tree["weighted_n_node_samples"]
                node = 0
                conditions: List[Dict[str, Any]] = []
                while int(left[node]) != int(right[node]):
                    feature_index = int(feature[node])
                    go_left = values[feature_index] <= float(threshold[node])
                    child = int(left[node] if go_left else right[node])
                    other = int(right[node] if go_left else left[node])
                    parent_mass = float(weighted[node])
                    decrease = (
                        parent_mass * float(impurity[node])
                        - float(weighted[int(left[node])]) * float(impurity[int(left[node])])
                        - float(weighted[int(right[node])]) * float(impurity[int(right[node])])
                    )
                    conditions.append(
                        {
                            "node_id": int(node),
                            "feature_index": feature_index,
                            "feature_name": tree["feature_names"][feature_index],
                            "feature_group": feature_group_from_name(
                                tree["feature_names"][feature_index]
                            ),
                            "observed_value": float(values[feature_index]),
                            "operator": "<=" if go_left else ">",
                            "threshold": float(threshold[node]),
                            "impurity_reduction": float(max(decrease, 0.0)),
                            "selected_child": child,
                            "unselected_child": other,
                        }
                    )
                    node = child
                prediction = int(tree["predicted_class"][node])
                output[row_index, user_index] = prediction
                total_decrease = sum(item["impurity_reduction"] for item in conditions)
                for condition in conditions:
                    condition["normalized_path_importance"] = (
                        condition["impurity_reduction"] / total_decrease
                        if total_decrease > 0.0
                        else 0.0
                    )
                sample_traces.append(
                    {
                        "user_index": int(user_index),
                        "chain_position": int(chain_position),
                        "preceding_users": [int(value) for value in preceding],
                        "predicted_class": prediction,
                        "leaf_node": int(node),
                        "conditions": conditions,
                        "path_importance_sum": float(
                            sum(item["normalized_path_importance"] for item in conditions)
                        ),
                    }
                )
            all_traces.append(sample_traces)
        return all_traces

    def to_rule_artifact(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        return {
            "format": MODEL_FORMAT,
            "numeric_semantics": RULE_EXECUTION_NUMERIC_SEMANTICS,
            "n_users": self.n_users,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "chain_enabled": self.chain_enabled,
            "chain_order": list(self.chain_order),
            "enabled_feature_groups": list(self.enabled_feature_groups),
            "class_weight": self.class_weight,
            "chain_training": self.chain_training,
            "oof_folds": self.oof_folds,
            "all_feature_names": list(self.all_feature_names),
            "feature_names": list(self.feature_names),
            "selected_feature_indices": self.selected_feature_indices.astype(int).tolist(),
            "compiled_trees": self.compiled_trees,
            "rule_texts": self.rule_texts,
            "fit_metadata": self.fit_metadata,
            "metrics": self.metrics,
        }

    def save_rules_json(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_rule_artifact(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(str(temporary), str(destination))
        return destination

    @classmethod
    def load_rules_json(cls, path: Union[str, Path]) -> "RulePolicyModel":
        source = Path(path)
        artifact = json.loads(source.read_text(encoding="utf-8"))
        if artifact.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported rule artifact: {artifact.get('format')!r}")
        model = cls(
            n_users=artifact["n_users"],
            max_depth=artifact["max_depth"],
            min_samples_leaf=artifact["min_samples_leaf"],
            random_state=artifact["random_state"],
            chain_enabled=artifact["chain_enabled"],
            chain_order=artifact["chain_order"],
            enabled_feature_groups=artifact["enabled_feature_groups"],
            class_weight=artifact["class_weight"],
            chain_training=artifact["chain_training"],
            oof_folds=artifact["oof_folds"],
        )
        model.all_feature_names = list(artifact["all_feature_names"])
        model.feature_names = list(artifact["feature_names"])
        model.selected_feature_indices = np.asarray(
            artifact["selected_feature_indices"], dtype=np.int64
        )
        model.compiled_trees = list(artifact["compiled_trees"])
        model.rule_texts = list(artifact["rule_texts"])
        model.fit_metadata = dict(artifact["fit_metadata"])
        model.metrics = dict(artifact["metrics"])
        model.models = []
        model.is_fitted = True
        return model

    def save(self, path: Union[str, Path]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "format": MODEL_FORMAT,
                    "artifact": self.to_rule_artifact(),
                    "models": self.models,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(str(temporary), str(destination))
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RulePolicyModel":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported pickle model: {payload.get('format')!r}")
        artifact = payload["artifact"]
        model = cls(
            n_users=artifact["n_users"],
            max_depth=artifact["max_depth"],
            min_samples_leaf=artifact["min_samples_leaf"],
            random_state=artifact["random_state"],
            chain_enabled=artifact["chain_enabled"],
            chain_order=artifact["chain_order"],
            enabled_feature_groups=artifact["enabled_feature_groups"],
            class_weight=artifact["class_weight"],
            chain_training=artifact["chain_training"],
            oof_folds=artifact["oof_folds"],
        )
        model.all_feature_names = list(artifact["all_feature_names"])
        model.feature_names = list(artifact["feature_names"])
        model.selected_feature_indices = np.asarray(
            artifact["selected_feature_indices"], dtype=np.int64
        )
        model.compiled_trees = list(artifact["compiled_trees"])
        model.rule_texts = list(artifact["rule_texts"])
        model.fit_metadata = dict(artifact["fit_metadata"])
        model.metrics = dict(artifact["metrics"])
        model.models = list(payload["models"])
        model.is_fitted = True
        return model

    def format_rule_summary_lines(self) -> List[str]:
        lines = [
            f"format={MODEL_FORMAT}",
            f"n_users={self.n_users}",
            f"chain_enabled={self.chain_enabled}",
            f"chain_order_one_based={[value + 1 for value in self.chain_order]}",
            f"chain_training={self.chain_training}",
            f"oof_folds={self.oof_folds}",
            f"enabled_feature_groups={list(self.enabled_feature_groups)}",
            f"max_depth={self.max_depth}",
            f"min_samples_leaf={self.min_samples_leaf}",
            f"total_rule_count={self.metrics.get('total_rule_count')}",
            f"average_rule_length={self.metrics.get('average_rule_length')}",
        ]
        for user_rules in self.rule_texts:
            user = int(user_rules["user_index"]) + 1
            lines.append(f"UE{user}_rules_begin")
            for index, rule in enumerate(user_rules["rules"], 1):
                conditions = " AND ".join(
                    f"{condition['feature_name']} {condition['operator']} "
                    f"{float(condition['threshold']):.17g}"
                    for condition in rule["conditions"]
                ) or "TRUE"
                lines.append(
                    f"rule_{index}: IF {conditions} THEN m_{user}="
                    f"{rule['predicted_class']} (samples={rule['sample_count']})"
                )
            lines.append(f"UE{user}_rules_end")
        return lines


def _self_test() -> None:
    rng = np.random.RandomState(42)
    channels = rng.lognormal(mean=-13.0, sigma=1.0, size=(240, 6))
    scaled = channels * CHANNEL_SCALE
    modes = np.zeros((240, 6), dtype=np.int8)
    modes[:, 0] = scaled[:, 0] > np.median(scaled[:, 0])
    modes[:, 1] = (scaled[:, 1] > np.mean(scaled, axis=1)).astype(np.int8)
    modes[:, 2] = np.logical_xor(modes[:, 0], modes[:, 1]).astype(np.int8)
    modes[:, 3] = (scaled[:, 3] > scaled[:, 4]).astype(np.int8)
    modes[:, 4] = np.logical_or(modes[:, 2], modes[:, 3]).astype(np.int8)
    modes[:, 5] = np.logical_xor(modes[:, 4], modes[:, 0]).astype(np.int8)

    model = RulePolicyModel(
        n_users=6,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        chain_enabled=True,
        chain_order=named_chain_order(6, "forward"),
        chain_training="cross_fitted",
        oof_folds=5,
    )
    model.fit(channels[:180], modes[:180])
    validation = model.evaluate(channels[180:], modes[180:])

    output_dir = Path(__file__).resolve().parent / "logs" / "rule_model_selftest"
    json_path = model.save_rules_json(output_dir / "rule_model_selftest.json")
    pickle_path = model.save(output_dir / "rule_model_selftest.pkl")
    json_model = RulePolicyModel.load_rules_json(json_path)
    pickle_model = RulePolicyModel.load(pickle_path)
    expected = model.predict_from_rules(channels)
    if not np.array_equal(json_model.predict_from_rules(channels), expected):
        raise AssertionError("JSON controller changed predictions")
    if not np.array_equal(pickle_model.predict_from_rules(channels), expected):
        raise AssertionError("pickle controller changed predictions")
    traces = json_model.trace_paths(channels[:2])
    if len(traces) != 2 or any(len(sample) != 6 for sample in traces):
        raise AssertionError("path tracing returned the wrong shape")
    for sample in traces:
        for trace in sample:
            if trace["conditions"] and not np.isclose(
                trace["path_importance_sum"], 1.0, atol=1e-12
            ):
                raise AssertionError("normalized path importance does not sum to one")

    print("RulePolicyModel self-test: PASSED")
    print(f"  feature groups: {list(FEATURE_GROUPS)}")
    print(f"  selected features: {len(model.feature_names)}")
    print(f"  validation bitwise fidelity: {validation['bitwise_fidelity']:.6f}")
    print(f"  validation joint fidelity: {validation['joint_fidelity']:.6f}")
    print(f"  tree-rule serialization fidelity: "
          f"{validation['tree_rule_serialization_fidelity']['joint_fidelity']:.6f}")
    print(f"  total executable rules: {model.metrics['total_rule_count']}")
    print(f"  artifacts: {output_dir}")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "MODEL_FORMAT",
    "RULE_EXECUTION_NUMERIC_SEMANTICS",
    "FEATURE_GROUPS",
    "GLOBAL_FEATURE_NAMES",
    "get_available_feature_groups",
    "feature_group_from_name",
    "select_feature_indices",
    "build_channel_feature_matrix",
    "validate_chain_order",
    "named_chain_order",
    "RulePolicyModel",
]
