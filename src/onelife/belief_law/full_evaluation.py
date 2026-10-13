"""Shared full-state predictive measurement channel with countable support.

This Monte Carlo marginal score covers every canonical serialized state field.
Objects and chunks are joint tuple-valued fields, preserving births/deaths.
It is an explicitly approximate scoring channel, not a native law likelihood.
"""

import math
from collections import Counter
from functools import lru_cache

import numpy as np

from onelife.belief_law.crafter import freeze
from onelife.belief_law.evaluation import MatchedObservableModel
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.our_method.crafter.observable_extractor import ObservableExtractor


@lru_cache(maxsize=32768)
def universal_log_mass(value):
    """Normalized recursive distribution on finite typed values.

    Tuples have geometric lengths and recursively distributed children.
    Integers use two-sided geometric mass; finite floats have uniform bit mass.
    Distinct unseen values retain distinct probabilities, without OTHER bins.
    """
    if value is None:
        return math.log(0.1)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return -math.inf
        terms = []
        integral = isinstance(value, int) or value.is_integer()
        if integral:
            terms.append(math.log(0.2 / 3) - abs(int(value)) * math.log(2))
            if value in (0, 1):
                terms.append(math.log(0.1 / 2))  # bools compare equal to 0/1.
        try:
            as_float = float(value)
        except OverflowError:
            as_float = math.inf
        if math.isfinite(as_float) and as_float == value:
            # Fold the masses of equal int/float/bool observations, including
            # the two IEEE encodings of zero, rather than losing normalization.
            terms.append(
                math.log(0.1)
                - math.log(2 * 2047)
                - 52 * math.log(2)
                + (math.log(2) if value == 0 else 0)
            )
        return float(np.logaddexp.reduce(terms))
    if isinstance(value, str):
        return (
            math.log(0.2)
            - (len(value) + 1) * math.log(2)
            - len(value) * math.log(0x110000)
        )
    if isinstance(value, tuple):
        return (
            math.log(0.3)
            - (len(value) + 1) * math.log(2)
            + sum(universal_log_mass(item) for item in value)
        )
    raise TypeError(f"Unsupported observation value {type(value)}")


def full_observations(state):
    """Fixed schema: no next-candidate-dependent domains or field selection."""
    data = _gamestate_to_json(state)
    result = {}

    def flatten(path, value):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(path + "." + key, item)
        elif path == "materials":
            for x, column in enumerate(value):
                for y, item in enumerate(column):
                    result[f"materials.{x}.{y}"] = item
        else:
            result[path] = freeze(value)

    for key, value in data.items():
        flatten(key, value)
    return result


class FullStateScorer(MatchedObservableModel):
    """Identical open-support marginal channel for every compared model.

    p(v) = (1-epsilon)*empirical(v) + epsilon*Q(v).
    Q is a normalized recursive typed prior. Half its mass is persistence;
    its integer branch is centered on the input when that input is integer.
    Neither candidate values nor held-out truth train the channel.
    """

    def __init__(self, model, *, samples=128, seed=42, noise_logscore=-10.0):
        super().__init__(
            model,
            ObservableExtractor(),
            samples=samples,
            seed=seed,
            noise_logscore=noise_logscore,
        )
        if not 0 < self.noise_mass < 1:
            raise ValueError("Full-state mixture mass must be in (0,1)")
        self.full_statistics = {}
        self.fallback_field_scores = 0

    def observations(self, state):
        observation = full_observations(state)
        # The original WorldState.__eq__ excludes view and chunks. They remain
        # in generation/edit-distance diagnostics, but cannot affect ranking.
        observation.pop("view")
        observation.pop("chunks")
        return observation

    def predictive_statistics(self, state, action):
        key = self.key(state, action)
        if key not in self.full_statistics:
            stats = {}
            for index in range(self.samples):
                nxt = self.draw(state, action, self.draw_seed(key, "ranking", index))
                observation = self.observations(nxt)
                for name, value in observation.items():
                    stats.setdefault(name, Counter())[value] += 1
            self.full_statistics[key] = stats
        return self.full_statistics[key]

    @staticmethod
    def base_log_mass(value, previous):
        integer_input = isinstance(previous, int) and not isinstance(previous, bool)
        base = math.log(0.4 if integer_input else 0.5) + universal_log_mass(value)
        if integer_input and (
            isinstance(value, int)
            or isinstance(value, float)
            and math.isfinite(value)
            and value.is_integer()
        ):
            integer_branch = math.log(0.1 / 3) - abs(int(value) - previous) * math.log(
                2
            )
            base = float(np.logaddexp(base, integer_branch))
        return float(np.logaddexp(base, math.log(0.5))) if value == previous else base

    def evaluate_log_probability(self, state, action, next_state):
        stats = self.predictive_statistics(state, action)
        previous, target = self.observations(state), self.observations(next_state)
        if target.keys() != previous.keys() or stats.keys() != previous.keys():
            raise ValueError(
                "World layout/schema changed; full-state field universe is fixed per input"
            )
        score = 0.0
        for name, value in target.items():
            count = stats[name][value]
            self.fallback_field_scores += count == 0
            empirical = (
                math.log((1 - self.noise_mass) * count / self.samples)
                if count
                else -math.inf
            )
            noise = math.log(self.noise_mass) + self.base_log_mass(
                value, previous[name]
            )
            score += float(np.logaddexp(empirical, noise))
        self.scored_candidates += 1
        self.nonfinite_scores += not math.isfinite(score)
        if not math.isfinite(score):
            raise ValueError("Nonfinite full-state channel score")
        return score

    def diagnostics(self):
        return {
            "scored_candidates": self.scored_candidates,
            "nonfinite_scores": self.nonfinite_scores,
            "fallback_field_scores": self.fallback_field_scores,
            "sample_budget": self.samples,
            "input_banks": len(self.full_statistics),
        }
