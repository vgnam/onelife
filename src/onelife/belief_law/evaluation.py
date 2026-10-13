"""A shared observational scoring channel for frozen Crafter world models.

The native likelihoods are deliberately not used. Both models supply the same
number of posterior predictive samples, projected through the saved OneLife
observable extractor. Each marginal receives the same additive noise mass.
This is a Monte Carlo marginal benchmark, not the native joint law likelihood.
"""

import copy
import json
import random
from collections import Counter
from collections.abc import Hashable
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from math import exp, log
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from crafter_oo.state_export import WorldState

from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.our_method.crafter.observable_extractor import ObservableExtractor

ABSENT = ("__absent_entity_attribute__",)
OTHER = ("__outside_observable_domain__",)


class Sampler(Protocol):
    def sample_next_state(
        self, current_state: WorldState, action: str
    ) -> WorldState: ...


@dataclass
class PredictiveBank:
    observations: list[dict[str, Hashable]]


@contextmanager
def isolated_randomness(model: Sampler, seed: int):
    """Keep predictive sampling from changing evaluator distractors/shuffles."""
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
    )
    posterior = getattr(model, "posterior", None)
    previous_rng = posterior.rng if posterior is not None else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if posterior is not None:
            posterior.rng = np.random.default_rng(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if posterior is not None:
            posterior.rng = previous_rng


class MatchedObservableModel:
    """Score predictive marginals using identical domains and smoothing.

    For domain size D, p(y) = (count(y)/N + exp(noise_logscore)) /
        (1 + D*exp(noise_logscore)).
    Domains include an OTHER category. Dynamic entity attributes include ABSENT;
    missing entity existence is 0. Domain selection never consults held-out values.
    All non-observed fields (daylight, survival counters, chunks, etc.) are
    marginalized out by projection, rather than required to match exactly.
    """

    def __init__(
        self,
        model: Sampler,
        extractor: ObservableExtractor,
        *,
        samples: int = 128,
        seed: int = 42,
        noise_logscore: float = -10.0,
        banks: dict[str, PredictiveBank] | None = None,
    ):
        if samples < 1 or not np.isfinite(noise_logscore):
            raise ValueError("samples must be positive and noise_logscore finite")
        self.model, self.extractor = model, extractor
        self.samples, self.seed = samples, seed
        self.noise_mass = exp(noise_logscore)
        if self.noise_mass == 0 or not np.isfinite(self.noise_mass):
            raise ValueError("noise mass must be finite and positive")
        self.banks = banks if banks is not None else {}
        self.draw_counts: Counter[str] = Counter()
        self.statistics: dict[tuple[str, int], dict[str, Counter]] = {}
        self.domain_cache: dict[str, frozenset[Hashable]] = {}
        self.scored_candidates = 0
        self.nonfinite_scores = 0
        self.missing_observation_hits = 0
        self.outside_domain_hits = 0

    def observations(self, state: WorldState) -> dict[str, Hashable]:
        return {
            str(key): value
            for key, value in self.extractor.get_observed_outcomes(state).items()
        }

    def key(self, state: WorldState, action: str) -> str:
        # Random/event state is excluded; no held-out next state enters this key.
        data = _gamestate_to_json(state)
        payload = json.dumps([data, action], sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def draw_seed(self, key: str, stream: str, index: int) -> int:
        payload = f"{self.seed}:{key}:{stream}:{index}".encode()
        return int.from_bytes(sha256(payload).digest()[:4], "little")

    def draw(self, state: WorldState, action: str, seed: int) -> WorldState:
        with isolated_randomness(self.model, seed):
            return self.model.sample_next_state(copy.deepcopy(state), action)

    def sample_next_state(self, current_state: WorldState, action: str) -> WorldState:
        key = self.key(current_state, action)
        index = self.draw_counts[key]
        self.draw_counts[key] += 1
        return self.draw(
            current_state, action, self.draw_seed(key, "generation", index)
        )

    def predictive_statistics(
        self, state: WorldState, action: str
    ) -> dict[str, Counter]:
        key = self.key(state, action)
        bank = self.banks.setdefault(key, PredictiveBank([]))
        while len(bank.observations) < self.samples:
            index = len(bank.observations)
            nxt = self.draw(state, action, self.draw_seed(key, "ranking", index))
            bank.observations.append(self.observations(nxt))
        cache_key = key, self.samples
        if cache_key not in self.statistics:
            stats: dict[str, Counter] = {}
            for observation in bank.observations[: self.samples]:
                for name, value in observation.items():
                    stats.setdefault(name, Counter())[self.bucket(name, value)] += 1
            self.statistics[cache_key] = stats
        return self.statistics[cache_key]

    def domain(self, name: str) -> frozenset[Hashable]:
        if name not in self.domain_cache:
            e = self.extractor
            if name.startswith("player_position_") or (
                name.startswith("entity_") and "_position_" in name
            ):
                values = e.position_domain
            elif name == "player_health" or name.endswith("_health"):
                # Inventory health uses the inventory domain below.
                values = (
                    e.inventory_domain
                    if name.startswith("player_inventory_")
                    else e.health_domain
                )
            elif name.startswith("player_inventory_"):
                values = e.inventory_domain
            elif name.startswith("entity_count_"):
                values = e.entity_count_domain
            elif name.startswith("entity_exists_"):
                values = e.entity_existence_domain
            elif name.endswith("_cooldown"):
                values = e.zombie_cooldown_domain
            elif name == "facing_tile":
                values = e.material_domain
            else:
                raise ValueError(f"No shared domain declared for observable {name}")
            domain: set[Hashable] = {int(value) for value in values}
            domain.add(OTHER)
            if name.startswith("entity_") and not name.startswith(
                ("entity_count_", "entity_exists_")
            ):
                domain.add(ABSENT)
            if name == "facing_tile":
                domain.add(ABSENT)
            self.domain_cache[name] = frozenset(domain)
        return self.domain_cache[name]

    def missing(self, name: str) -> Hashable:
        return 0 if name.startswith("entity_exists_") else ABSENT

    def bucket(self, name: str, value: Hashable) -> Hashable:
        return value if value in self.domain(name) else OTHER

    def marginal_probability(
        self, name: str, counts: Counter, value: Hashable
    ) -> float:
        missing_count = self.samples - sum(counts.values())
        bucket = self.bucket(name, value)
        count = counts[bucket]
        if bucket == self.bucket(name, self.missing(name)):
            count += missing_count
        return (count / self.samples + self.noise_mass) / (
            1 + len(self.domain(name)) * self.noise_mass
        )

    def evaluate_log_probability(
        self, state: WorldState, action: str, next_state: WorldState
    ) -> float:
        stats = self.predictive_statistics(state, action)
        before, after = self.observations(state), self.observations(next_state)
        score = 0.0
        for name in sorted(before.keys() | after.keys() | stats.keys()):
            value = after.get(name, self.missing(name))
            self.missing_observation_hits += int(name not in after)
            self.outside_domain_hits += int(value not in self.domain(name))
            score += log(
                self.marginal_probability(name, stats.get(name, Counter()), value)
            )
        self.scored_candidates += 1
        self.nonfinite_scores += int(not np.isfinite(score))
        return score

    def diagnostics(self) -> dict:
        return {
            "samples_per_state_action": self.samples,
            "predictive_state_actions": len(self.banks),
            "scored_candidates": self.scored_candidates,
            "nonfinite_scores": self.nonfinite_scores,
            "missing_observation_hits": self.missing_observation_hits,
            "outside_domain_hits": self.outside_domain_hits,
            "scorer": "shared empirical marginal + additive noise",
            "noise_logscore": log(self.noise_mass),
        }

    def save_banks(self, path: Path) -> None:
        """Portable local cache; JSON encodes only integer observations."""
        path.write_text(
            json.dumps({key: bank.observations for key, bank in self.banks.items()}),
            encoding="utf-8",
        )

    def load_banks(self, path: Path) -> None:
        self.banks.update(
            {
                key: PredictiveBank(observations)
                for key, observations in json.loads(
                    path.read_text(encoding="utf-8")
                ).items()
            }
        )
