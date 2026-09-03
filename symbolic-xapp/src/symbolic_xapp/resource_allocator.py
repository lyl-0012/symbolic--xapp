"""Model-based continuous allocator for a fixed WP-MEC mode vector.

Only the allocator required by the proposed symbolic-control pipeline is
included. Baseline search and comparison algorithms are intentionally absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import lambertw


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


@dataclass(frozen=True)
class SystemParameters:
    """Physical parameters shared by all experiments."""

    cycles_per_bit: float = 100.0          # phi in the manuscript
    downlink_power_w: float = 3.0          # P
    energy_efficiency: float = 0.7         # eta
    switched_capacitance: float = 1.0e-26  # kappa
    receiver_noise_w: float = 1.0e-10      # N_0
    bandwidth_hz: float = 2.0e6            # B
    uplink_overhead: float = 1.1            # v_u

    def validate(self) -> None:
        values = asdict(self)
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError(f"system parameters must be finite: {values}")
        if any(value <= 0.0 for value in values.values()):
            raise ValueError(f"system parameters must be positive: {values}")
        if self.energy_efficiency > 1.0:
            raise ValueError("energy_efficiency must not exceed one")


DEFAULT_PARAMETERS = SystemParameters()
DEFAULT_BISECTION_DELTA = 0.005
NUMERICAL_TOLERANCE = 1.0e-9


def default_service_weights(n_users: int) -> np.ndarray:
    """Return [1, 1.5, 1, 1.5, ...] for zero-based UE indices."""

    n_users = int(n_users)
    if n_users <= 0:
        raise ValueError("n_users must be positive")
    weights = np.ones(n_users, dtype=np.float64)
    weights[1::2] = 1.5
    return weights


def _validate_inputs(
    h: ArrayLike,
    mode: ArrayLike,
    weights: Optional[ArrayLike],
    parameters: SystemParameters,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    parameters.validate()
    if not np.isfinite(delta) or float(delta) <= 0.0:
        raise ValueError(f"delta must be positive and finite, got {delta}")

    channel = np.asarray(h, dtype=np.float64).reshape(-1)
    binary_mode = np.asarray(mode).reshape(-1)
    if channel.size == 0:
        raise ValueError("channel vector must not be empty")
    if binary_mode.shape != channel.shape:
        raise ValueError(
            f"mode shape {binary_mode.shape} does not match channel shape {channel.shape}"
        )
    if not np.all(np.isfinite(channel)):
        raise ValueError("channel vector contains NaN or infinity")
    if np.any(channel <= 0.0):
        raise ValueError("channel power gains must be strictly positive")
    if not np.all(np.logical_or(binary_mode == 0, binary_mode == 1)):
        raise ValueError("mode vector must be binary")
    binary_mode = binary_mode.astype(np.int8, copy=False)

    if weights is None or np.asarray(weights).size == 0:
        service_weights = default_service_weights(channel.size)
    else:
        service_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if service_weights.shape != channel.shape:
            raise ValueError(
                f"weight shape {service_weights.shape} does not match "
                f"channel shape {channel.shape}"
            )
        if not np.all(np.isfinite(service_weights)) or np.any(service_weights <= 0.0):
            raise ValueError("service weights must be positive and finite")
    return channel, binary_mode, service_weights


def _derived_constants(parameters: SystemParameters) -> Tuple[float, float, float]:
    eta_local = (
        (parameters.energy_efficiency * parameters.downlink_power_w) ** (1.0 / 3.0)
        / parameters.cycles_per_bit
    )
    eta_offload = (
        parameters.energy_efficiency
        * parameters.downlink_power_w
        / parameters.receiver_noise_w
    )
    spectral_scale = parameters.bandwidth_hz / (
        parameters.uplink_overhead * np.log(2.0)
    )
    return float(eta_local), float(eta_offload), float(spectral_scale)


def _offloading_rate(
    tau: np.ndarray,
    channel: np.ndarray,
    energy_fraction: float,
    eta_offload: float,
    spectral_scale: float,
) -> np.ndarray:
    """Evaluate tau*log(1+c/tau), including its zero-tau limit."""

    tau = np.asarray(tau, dtype=np.float64)
    result = np.zeros_like(tau)
    positive = tau > 0.0
    if np.any(positive):
        numerator = eta_offload * np.square(channel[positive]) * energy_fraction
        result[positive] = (
            spectral_scale
            * tau[positive]
            * np.log1p(numerator / tau[positive])
        )
    return result


def bisection(
    h: ArrayLike,
    M: ArrayLike,
    weights: Optional[ArrayLike] = None,
    return_details: bool = False,
    delta: float = DEFAULT_BISECTION_DELTA,
    parameters: SystemParameters = DEFAULT_PARAMETERS,
    max_bisection_iterations: int = 256,
) -> Any:
    """Solve continuous allocation for one fixed binary mode vector.

    Returns
    -------
    gain, a, tau_offload
        Backward-compatible return. ``tau_offload`` contains entries only for
        UEs with mode 1, in ascending UE-index order.
    gain, a, tau_offload, details
        Returned when ``return_details=True``. ``details['tau_full']`` always
        has length N and is the preferred representation for saved results.
    """

    channel, mode, service_weights = _validate_inputs(
        h, M, weights, parameters, delta
    )
    n_users = channel.size
    local_indices = np.flatnonzero(mode == 0)
    offload_indices = np.flatnonzero(mode == 1)
    eta_local, eta_offload, spectral_scale = _derived_constants(parameters)

    local_h = channel[local_indices]
    offload_h = channel[offload_indices]
    local_w = service_weights[local_indices]
    offload_w = service_weights[offload_indices]

    local_coefficient = float(
        np.sum(
            local_w
            * eta_local
            * np.power(local_h / parameters.switched_capacitance, 1.0 / 3.0)
        )
    )

    # With no offloading users, the objective is monotone in a and a*=1.
    if offload_indices.size == 0:
        dual_value = local_coefficient / 3.0
        energy_fraction = 1.0
        tau_offload = np.empty(0, dtype=np.float64)
        bisection_iterations = 0
        bracket_expansions = 0
    else:

        def phi_vector(v: float) -> np.ndarray:
            exponent = -1.0 - v / (offload_w * spectral_scale)
            argument = -np.exp(exponent)
            principal_w = np.real(lambertw(argument, k=0))
            result = np.zeros_like(principal_w, dtype=np.float64)
            nonzero = np.abs(principal_w) > np.finfo(np.float64).tiny
            denominator = -1.0 - 1.0 / principal_w[nonzero]
            result[nonzero] = 1.0 / denominator
            # Roundoff near the branch point may produce tiny negative values.
            return np.maximum(result, 0.0)

        def allocation_from_dual(v: float) -> Tuple[float, np.ndarray]:
            phi_values = phi_vector(v)
            weighted_sum = float(np.sum(np.square(offload_h) * phi_values))
            a_value = 1.0 / (1.0 + eta_offload * weighted_sum)
            tau_values = (
                eta_offload * np.square(offload_h) * a_value * phi_values
            )
            return float(a_value), np.asarray(tau_values, dtype=np.float64)

        def stationarity(v: float) -> float:
            a_value, tau_values = allocation_from_dual(v)
            phi_values = np.zeros_like(tau_values)
            positive_tau = tau_values > 0.0
            if np.any(positive_tau):
                phi_values[positive_tau] = (
                    tau_values[positive_tau]
                    / (
                        eta_offload
                        * np.square(offload_h[positive_tau])
                        * a_value
                    )
                )
            local_term = (
                0.0
                if local_coefficient == 0.0
                else local_coefficient * np.power(a_value, -2.0 / 3.0) / 3.0
            )
            offload_term = float(
                np.sum(
                    offload_w
                    * np.square(offload_h)
                    / (1.0 + np.divide(1.0, phi_values, out=np.full_like(phi_values, np.inf), where=phi_values > 0.0))
                )
            )
            return local_term + spectral_scale * eta_offload * offload_term - v

        lower = 0.0
        upper = 1.0
        bracket_expansions = 0
        upper_value = stationarity(upper)
        while upper_value > 0.0:
            upper *= 2.0
            bracket_expansions += 1
            if bracket_expansions > 80 or not np.isfinite(upper):
                raise RuntimeError("failed to bracket the allocation dual variable")
            upper_value = stationarity(upper)
            if not np.isfinite(upper_value):
                raise FloatingPointError(
                    f"non-finite stationarity value while bracketing: {upper_value}"
                )

        bisection_iterations = 0
        while upper - lower > float(delta):
            dual_value = 0.5 * (lower + upper)
            value = stationarity(dual_value)
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"non-finite stationarity value during bisection: {value}"
                )
            if value > 0.0:
                lower = dual_value
            else:
                upper = dual_value
            bisection_iterations += 1
            if bisection_iterations >= int(max_bisection_iterations):
                raise RuntimeError(
                    "bisection failed to converge within "
                    f"{max_bisection_iterations} iterations"
                )

        dual_value = 0.5 * (lower + upper)
        energy_fraction, tau_offload = allocation_from_dual(dual_value)

    tau_full = np.zeros(n_users, dtype=np.float64)
    tau_full[offload_indices] = tau_offload

    local_rates = np.zeros(n_users, dtype=np.float64)
    if local_indices.size:
        local_rates[local_indices] = (
            eta_local
            * np.power(
                channel[local_indices] / parameters.switched_capacitance,
                1.0 / 3.0,
            )
            * np.power(energy_fraction, 1.0 / 3.0)
        )

    offload_rates = np.zeros(n_users, dtype=np.float64)
    if offload_indices.size:
        offload_rates[offload_indices] = _offloading_rate(
            tau_offload,
            offload_h,
            energy_fraction,
            eta_offload,
            spectral_scale,
        )

    per_user_rates = local_rates + offload_rates
    gain = float(np.dot(service_weights, per_user_rates))
    time_sum = float(energy_fraction + np.sum(tau_full))
    time_residual = float(1.0 - time_sum)

    if energy_fraction < -NUMERICAL_TOLERANCE or energy_fraction > 1.0 + NUMERICAL_TOLERANCE:
        raise RuntimeError(f"invalid energy-transfer fraction: {energy_fraction}")
    if np.any(tau_full < -NUMERICAL_TOLERANCE):
        raise RuntimeError("negative uplink allocation returned by bisection")
    if time_sum > 1.0 + 1.0e-7:
        raise RuntimeError(f"time constraint violated: a+sum(tau)={time_sum}")
    if np.any(tau_full[local_indices] != 0.0):
        raise RuntimeError("local-computing UE received nonzero uplink time")
    if not np.isfinite(gain) or gain < 0.0:
        raise RuntimeError(f"invalid weighted computation rate: {gain}")

    if not return_details:
        return gain, float(energy_fraction), tau_offload.copy()

    details: Dict[str, Any] = {
        "mode": mode.astype(int, copy=True),
        "local_indices": local_indices.astype(int, copy=True),
        "offload_indices": offload_indices.astype(int, copy=True),
        "tau_full": tau_full,
        "local_rates": local_rates,
        "offload_rates": offload_rates,
        "per_user_rates": per_user_rates,
        "weights": service_weights.copy(),
        "weighted_per_user_rates": service_weights * per_user_rates,
        "weighted_sum_rate": gain,
        "energy_transfer_fraction": float(energy_fraction),
        "time_sum": time_sum,
        "time_constraint_residual": time_residual,
        "dual_value": float(dual_value),
        "bisection_delta": float(delta),
        "bisection_iterations": int(bisection_iterations),
        "bracket_expansions": int(bracket_expansions),
        "system_parameters": asdict(parameters),
    }
    return gain, float(energy_fraction), tau_offload.copy(), details


__all__ = [
    "SystemParameters",
    "DEFAULT_PARAMETERS",
    "DEFAULT_BISECTION_DELTA",
    "default_service_weights",
    "bisection",
]
