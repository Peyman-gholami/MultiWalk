"""Per-client, per-round participation probabilities for federated algorithms.

Patterns match the common experimental regimes:
  - uniform:           fixed fraction via without-replacement sampling (legacy)
  - stationary:        p_i^r = high_i  (Bernoulli with constant per-client rate)
  - staircase:         square wave between high_i/low_i with phase shift across clients
  - sine:              smooth oscillation in [low_i, high_i]
  - interleaved_sine:  half-sine pulses separated by inactivity (zeros); peak = high_i

`participation_rate` / `participation_low` may be a single float (shared by all
clients) or a list of length N (one value per client, client index 0..N-1).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch


PARTICIPATION_PATTERNS = (
    "uniform",
    "stationary",
    "staircase",
    "sine",
    "interleaved_sine",
)

FloatOrList = Union[float, Sequence[float]]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _client_phase(client_index: int, num_clients: int) -> float:
    """Evenly spaced phases so clients are interleaved over a cycle."""
    if num_clients <= 1:
        return 0.0
    return 2.0 * math.pi * client_index / num_clients


def _as_client_values(value: FloatOrList, num_clients: int, name: str) -> List[float]:
    """Broadcast a scalar (or length-1 list) or validate a per-client list."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [float(value[0])] * num_clients
        if len(value) != num_clients:
            raise ValueError(
                f"{name} has length {len(value)}, but there are {num_clients} clients. "
                f"Pass a single float or exactly {num_clients} values."
            )
        return [float(v) for v in value]
    return [float(value)] * num_clients


def participation_probability(
    client_index: int,
    round_number: int,
    num_clients: int,
    pattern: str,
    high: float,
    low: float = 0.0,
    period: int = 50,
) -> float:
    """Return p_i^r in [0, 1] for client index i in {0, ..., N-1} at round r.

    `high` / `low` here are already the per-client scalars high_i / low_i.
    """
    pattern = pattern.lower()
    high = _clamp01(high)
    low = _clamp01(low)
    if low > high:
        low, high = high, low
    period = max(1, int(period))
    phase = _client_phase(client_index, num_clients)
    # Progress in [0, 1) over one period, with per-client phase offset.
    phase_rounds = phase / (2.0 * math.pi) * period
    progress = ((round_number + phase_rounds) % period) / period

    if pattern in ("uniform", "stationary"):
        # Stationary: constant in time for this client.
        # Uniform uses fixed-count sampling in select_participating_clients.
        return high

    if pattern == "staircase":
        # Square wave: first half of (phase-shifted) period at high_i, second at low_i.
        return high if progress < 0.5 else low

    if pattern == "sine":
        # Smooth oscillation between low_i and high_i.
        mid = 0.5 * (high + low)
        amp = 0.5 * (high - low)
        return _clamp01(mid + amp * math.sin(2.0 * math.pi * progress))

    if pattern == "interleaved_sine":
        # Half-sine bumps with inactivity between them; peak is high_i.
        wave = math.sin(2.0 * math.pi * progress)
        return _clamp01(high * max(0.0, wave))

    raise ValueError(
        f"Unknown participation pattern '{pattern}'. "
        f"Expected one of {PARTICIPATION_PATTERNS}."
    )


def participation_probabilities(
    round_number: int,
    num_clients: int,
    pattern: str,
    high: FloatOrList,
    low: FloatOrList = 0.0,
    period: int = 50,
) -> List[float]:
    highs = _as_client_values(high, num_clients, "participation_rate")
    lows = _as_client_values(low, num_clients, "participation_low")
    return [
        participation_probability(
            i, round_number, num_clients, pattern, highs[i], lows[i], period
        )
        for i in range(num_clients)
    ]


def select_participating_clients(
    config: dict,
    round_number: int,
    total_clients: int,
    *,
    ensure_at_least_one: bool = True,
) -> List[int]:
    """Sample participating client ranks (1..N) for the current round.

    Config keys:
      participation_pattern: one of PARTICIPATION_PATTERNS (default: uniform)
      participation_rate:    high_i — float or list of length N (default: 1.0)
      participation_low:     low_i  — float or list of length N (default: 0.0)
      participation_period:  cycle length in rounds (default: 50)
      seed:                  RNG seed
    """
    pattern = str(config.get("participation_pattern", "uniform")).lower()
    high = config.get("participation_rate", 1.0)
    low = config.get("participation_low", 0.0)
    period = int(config.get("participation_period", 50))
    seed = int(config.get("seed", 0))

    torch.manual_seed(seed + round_number)

    # Legacy behavior: pick a fixed fraction without replacement.
    # If a list is given, use the mean as the target fraction.
    if pattern == "uniform":
        highs = _as_client_values(high, total_clients, "participation_rate")
        fraction = sum(highs) / max(1, len(highs))
        num_participants = max(1, int(fraction * total_clients)) if ensure_at_least_one else int(fraction * total_clients)
        num_participants = min(total_clients, max(0, num_participants))
        if num_participants == 0:
            return []
        indices = torch.randperm(total_clients)[:num_participants].tolist()
        return [idx + 1 for idx in indices]

    probs = participation_probabilities(round_number, total_clients, pattern, high, low, period)
    draws = torch.rand(total_clients)
    selected = [i + 1 for i, (p, u) in enumerate(zip(probs, draws.tolist())) if u < p]

    if ensure_at_least_one and not selected and total_clients > 0:
        # Fall back to the client with highest p_i^r this round (ties broken randomly).
        scores = torch.tensor(probs, dtype=torch.float32)
        scores = scores + 1e-6 * torch.rand(total_clients)
        selected = [int(torch.argmax(scores).item()) + 1]

    return selected
