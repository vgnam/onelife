"""Local, train-only library revision and structural Crafter decoding.

No simulator transitions or held-out successors are used for induction. This
is a finite library revision, not an implementation of unrestricted synthesis.
"""

import copy
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from typing import cast

import numpy as np
from scipy.optimize import least_squares

from onelife.belief_law.core import (
    Candidate,
    CategoricalCatchAll,
    Outcome,
    changed,
    equal,
    matches,
)
from onelife.belief_law.crafter import CrafterAdapter
from onelife.our_method.action_remapping import remap_slug_actions_to_balrog_actions


class BackoffCatchAll(CategoricalCatchAll):
    """Joint portable edits with pooled train support for unseen contexts.

    Pooling is an acquisition prior, while context counts remain prequential.
    It never factors a birth into independent field draws.
    """

    integrated_probabilities = True

    def predict(self, state, action):
        ctx = self.context(state, action)
        support = sorted(
            self.support.get(ctx) or set().union({()}, *self.support.values()), key=repr
        )
        counts = self.counts.get(ctx, Counter())
        pooled = sum(self.counts.values(), Counter())
        prior_total = sum(pooled[s] for s in support) + self.beta * len(support)
        prior = {s: (pooled[s] + self.beta) / prior_total for s in support}
        denominator = sum(counts.values()) + 1.0
        persistence = {k: v for k, v in state.items() if matches(k, self.writes)}
        return [
            Outcome(
                {**persistence, **self.restore(state, s)},
                (counts[s] + prior[s]) / denominator,
            )
            for s in support
        ]


class OpenScalarCatchAll(BackoffCatchAll):
    """A normalized countable escape branch for an isolated numeric slot.

    Integer escapes follow a two-sided geometric distribution. Daylight uses
    the finite set of nonnegative IEEE-754 doubles in [0, 1], uniformly in bit
    representation. This is probability mass, not a continuous density added
    to a discrete law. Sampling and analytic scoring use the same mixture.
    """

    escape_mass = 0.01
    float_max_bits = struct.unpack("Q", struct.pack("d", 1.0))[0]

    def __init__(self, *args, key, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key

    def base_probability(self, state, value):
        if self.key == "world.daylight":
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return 0.0
            return 1.0 / (self.float_max_bits + 1)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            return 0.0
        delta = abs(int(value) - int(state[self.key]))
        return math.exp(-math.log(3) - delta * math.log(2))

    def base_sample(self, state):
        if self.key == "world.daylight":
            bits = int(np.random.randint(0, self.float_max_bits + 1, dtype=np.int64))
            return struct.unpack("d", struct.pack("Q", bits))[0]
        if np.random.random() < 1 / 3:
            delta = 0
        else:
            delta = int(np.random.geometric(0.5)) * int(np.random.choice([-1, 1]))
        return int(state[self.key]) + delta

    def predict(self, state, action):
        result = [
            Outcome(o.edits, (1 - self.escape_mass) * o.probability)
            for o in super().predict(state, action)
        ]
        return result + [Outcome({self.key: self.base_sample(state)}, self.escape_mass)]

    def predict_observed(self, state, action, observed):
        result = [
            Outcome(o.edits, (1 - self.escape_mass) * o.probability)
            for o in super().predict(state, action)
        ]
        value = observed.get(self.key)
        return result + [
            Outcome(
                {self.key: value},
                self.escape_mass * self.base_probability(state, value),
            )
        ]


@dataclass
class PeriodicLaw:
    period: float
    phase: float
    power: float
    offset: float
    amplitude: float
    increment: int
    cosine: str = "numpy"

    def value(self, step):
        cosine = np.cos if self.cosine == "numpy" else math.cos
        return float(
            self.offset
            + self.amplitude
            * abs(cosine(np.pi * ((step / self.period) % 1 + self.phase))) ** self.power
        )

    def predict(self, state, action):
        return [
            Outcome(
                {
                    "world.daylight": self.value(
                        int(state["world.step_count"]) + self.increment
                    )
                }
            )
        ]


def infer_clock(history):
    increments = Counter(
        int(n["world.step_count"]) - int(s["world.step_count"]) for s, _, n in history
    )
    if len(increments) != 1:
        raise ValueError("Train data does not support a constant clock increment")
    increment = next(iter(increments))
    values = {
        float(s["world.step_count"]): float(s["world.daylight"])
        for s, _, n in history
        for s in (s, n)
    }
    x = np.array(sorted(values))
    y = np.array([values[v] for v in x])

    def residual(parameters):
        period, phase, power, offset, amplitude = parameters
        return (
            offset
            + amplitude * np.abs(np.cos(np.pi * ((x / period) % 1 + phase))) ** power
            - y
        )

    solutions = []
    # Generic periodic powered-cosine family; parameters are fitted to train.
    for period in (100, 200, 400, 800):
        for phase, amplitude in (
            (-0.5, 1),
            (0, 1),
            (0.5, 1),
            (-0.5, -1),
            (0, -1),
            (0.5, -1),
        ):
            fit = least_squares(
                residual,
                [period, phase, 2, 0 if amplitude > 0 else 1, amplitude],
                bounds=([20, -2, 0.2, -1, -2], [10000, 2, 8, 1.1, 2]),
                max_nfev=1500,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
            )
            solutions.append((float(np.max(np.abs(residual(fit.x)))), fit.x))
    error, parameters = min(solutions, key=lambda item: item[0])
    snapped = np.array(
        [float(Fraction(float(v)).limit_denominator(1000)) for v in parameters]
    )
    if np.max(np.abs(residual(snapped))) < 1e-10:
        parameters = snapped
    if error > 1e-8:
        raise ValueError(f"Periodic daylight family failed train validation: {error}")
    alternatives = [
        PeriodicLaw(
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
            float(parameters[4]),
            increment,
            cosine,
        )
        for cosine in ("numpy", "math")
    ]
    law = min(
        alternatives,
        key=lambda law: (
            sum(law.value(step) != value for step, value in values.items()),
            max(abs(law.value(step) - value) for step, value in values.items()),
        ),
    )
    return law, {
        "increment": increment,
        "parameters": parameters.tolist(),
        "max_train_error": max(abs(law.value(k) - v) for k, v in values.items()),
        "exact_float_matches": sum(law.value(k) == v for k, v in values.items()),
        "train_clock_states": len(values),
        "cosine": law.cosine,
    }


@dataclass
class GuardTree:
    labels: list[tuple]
    features: list[str]
    root: dict

    def predict(self, state, action):
        node = self.root
        while "feature" in node:
            value = (
                action if node["feature"] == "__action__" else state[node["feature"]]
            )
            left = (
                value <= node["threshold"]
                if node["numeric"]
                else value == node["threshold"]
            )
            node = node["left"] if left else node["right"]
        counts = node["counts"]
        total = sum(counts) + 0.05 * len(self.labels)
        outcomes = []
        for label, count in zip(self.labels, counts):
            edits = {k: state[k] + v if mode == "delta" else v for k, mode, v in label}
            outcomes.append(Outcome(edits, (count + 0.05) / total))
        return outcomes


def infer_survival(history, max_depth):
    fields = ["player." + k for k in ("hunger", "thirst", "fatigue", "recover")]
    features = sorted(
        k
        for k, v in history[0][0].items()
        if k.startswith("player.") and isinstance(v, (int, float, bool))
    )
    features += ["__action__"]
    labels, rows = [], []
    for state, action, nxt in history:
        label = tuple(
            (k, "set", 0)
            if nxt[k] == 0 and state[k] != 0
            else (k, "delta", nxt[k] - state[k])
            for k in fields
        )
        if label not in labels:
            labels.append(label)
        rows.append(
            (
                [action if k == "__action__" else state[k] for k in features],
                labels.index(label),
            )
        )

    def impurity(indices):
        counts = Counter(rows[i][1] for i in indices)
        return len(indices) - sum(c * c for c in counts.values()) / len(indices)

    def build(indices, depth):
        counts = Counter(rows[i][1] for i in indices)
        leaf = {"counts": [counts[j] for j in range(len(labels))]}
        if depth == max_depth or len(counts) == 1 or len(indices) < 4:
            return leaf
        best = None
        for f, feature in enumerate(features):
            values = sorted({rows[i][0][f] for i in indices})
            numeric = feature != "__action__" and not isinstance(values[0], bool)
            thresholds = (
                [(a + b) / 2 for a, b in pairwise(values)] if numeric else values
            )
            for threshold in thresholds:
                left = [
                    i
                    for i in indices
                    if (
                        rows[i][0][f] <= threshold
                        if numeric
                        else rows[i][0][f] == threshold
                    )
                ]
                if not left or len(left) == len(indices):
                    continue
                left_set = set(left)
                right = [i for i in indices if i not in left_set]
                loss = impurity(left) + impurity(right)
                if best is None or loss < best[0]:
                    best = (loss, feature, threshold, numeric, left, right)
        if best is None or best[0] >= impurity(indices) - 1e-8:
            return leaf
        _, feature, threshold, numeric, left, right = best
        return {
            "feature": feature,
            "threshold": threshold,
            "numeric": numeric,
            "left": build(left, depth + 1),
            "right": build(right, depth + 1),
        }

    return GuardTree(labels, features, build(list(range(len(rows))), 0))


class StructuralAdapter(CrafterAdapter):
    """Complete births using train prototypes before emitting a WorldState.

    A partial removal removes the identity as a unit. This declared projection
    is scored through the emitted-state sample bank, not the old symbolic PMF.
    """

    def __init__(self, original, history):
        super().__init__(original.template, original.outcome_budget)
        self.prototypes = {}
        self._decoded = original._decoded
        for before, _, after in history:
            for state in (before, after):
                prefixes = {
                    ".".join(k.split(".")[:3]) for k in state if k.startswith("entity.")
                }
                for prefix in sorted(prefixes):
                    kind = prefix.split(".")[1]
                    fields = {
                        k[len(prefix) + 1 :]: v
                        for k, v in state.items()
                        if k.startswith(prefix + ".")
                    }
                    self.prototypes.setdefault(kind, fields)

    def signature(self, state, edits):
        signature = cast(tuple, super().signature(state, edits))
        new = sorted(
            {
                ".".join(k.split(".")[:3])
                for k in edits
                if k.startswith("entity.")
                and ".".join(k.split(".")[:3]) not in self._entity_map(state)
            },
            key=lambda k: int(k.split(".")[2]),
        )
        offsets = {
            p: int(p.split(".")[2]) - int(state["world.entity_id_counter_state"])
            for p in new
        }
        mapping = {p: f"entity.{p.split('.')[1]}.@new{offsets[p]}" for p in new}
        result = []
        for key, mode, value in signature:
            prefix = ".".join(key.split(".")[:3])
            if prefix in mapping:
                key = mapping[prefix] + key[len(prefix) :]
                if key.endswith(".entity_id"):
                    value = ("__new_entity_id__", offsets[prefix])
            result.append((key, mode, value))
        return tuple(result)

    def restore(self, state, signature):
        converted = []
        counter = int(state["world.entity_id_counter_state"])
        for key, mode, value in signature:
            if key.startswith("entity.") and ".@new" in key:
                parts = key.split(".")
                rank = int(parts[2][4:])
                parts[2] = str(counter + rank)
                key = ".".join(parts)
                if (
                    isinstance(value, tuple)
                    and value
                    and value[0] == "__new_entity_id__"
                ):
                    value = counter + rank
            converted.append((key, mode, value))
        return super().restore(state, tuple(converted))

    def decode(self, state, reference=None):
        state = dict(state)
        previous = self.encode(reference or self.template)
        prefixes = {
            ".".join(k.split(".")[:3]) for k in state if k.startswith("entity.")
        }
        for prefix in sorted(prefixes):
            fields = {
                k[len(prefix) + 1 :]: v
                for k, v in state.items()
                if k.startswith(prefix + ".")
            }
            existed = prefix + ".name" in previous
            if existed and ("name" not in fields or "entity_id" not in fields):
                state = {
                    k: v for k, v in state.items() if not k.startswith(prefix + ".")
                }
                self.approximations["atomic_entity_removals"] += 1
                continue
            if not existed:
                kind, eid = prefix.split(".")[1:]
                prototype = self.prototypes.get(kind)
                required = {
                    "name",
                    "entity_id",
                    "position.x",
                    "position.y",
                    "health",
                } | {
                    "zombie": {"cooldown"},
                    "skeleton": {"reload"},
                    "plant": {"grown", "ripe"},
                    "arrow": {"facing.x", "facing.y"},
                }.get(kind, set())
                if prototype is None and not required.issubset(fields):
                    raise ValueError(f"No train birth prototype for {kind}")
                for field, value in (prototype or {}).items():
                    state.setdefault(prefix + "." + field, value)
                state[prefix + ".name"] = kind
                state[prefix + ".entity_id"] = int(eid)
                if not required.issubset(fields):
                    self.approximations["completed_births_from_train_prototype"] += 1
        result = super().decode(state, reference)
        ids = [o.entity_id for o in result.objects]
        result.entity_id_counter_state = max(
            result.entity_id_counter_state, max(ids, default=-1) + 1
        )
        # Chunk membership is a schema invariant: preserve existing empty
        # chunks, rebuild memberships from the emitted entity positions.
        memberships = defaultdict(list)
        for obj in result.objects:
            x, y = obj.position.x, obj.position.y
            w, h = result.chunk_size
            left, top = x // w * w, y // h * h
            memberships[(left, left + w, top, top + h)].append(obj.entity_id)
        from crafter_oo.state_export import ChunkState

        keys = {tuple(chunk.chunk_key) for chunk in result.chunks} | memberships.keys()
        result.chunks = [
            ChunkState(chunk_key=key, objects=sorted(memberships[key]))
            for key in sorted(keys)
        ]
        return result


def revise_locally(posterior, original):
    history = copy.deepcopy(posterior.history)
    adapter = StructuralAdapter(original, history)
    clock, clock_report = infer_clock(history)
    events = []
    for i, slot in enumerate(posterior.slots):
        table = posterior.catchalls[i]
        scalar = len(slot.writes) == 1 and next(iter(slot.writes)) in {
            "world.step_count",
            "world.daylight",
            "world.entity_id_counter_state",
        }
        cls = OpenScalarCatchAll if scalar else BackoffCatchAll
        kwargs = {"key": next(iter(slot.writes))} if scalar else {}
        replacement = cls(
            table.writes,
            table.context,
            adapter.signature,
            adapter.restore,
            table.beta,
            **kwargs,
        )
        posterior.catchalls[i] = replacement
        for state, action, nxt in history:
            edits = {
                k: v for k, v in changed(state, nxt).items() if matches(k, slot.writes)
            }
            restored = replacement.restore(state, replacement.signature(state, edits))
            if any(not equal(restored.get(k), v) for k, v in edits.items()):
                raise ValueError(f"Train catch-all roundtrip failed in slot {i}")
        for c in slot.candidates:
            if c.kind == "catchall":
                c.predict = replacement.predict
                # Pooled and escape probabilities are integrated means, rather
                # than incorrectly using the old context Dirichlet vector.
            elif c.kind == "program":
                # Existing closures keep the old adapter; it has no change to
                # program effects. New catch-alls use portable birth identities.
                pass
        if slot.writes == frozenset({"world.step_count"}):
            increment = clock.increment
            slot.candidates.append(
                Candidate(
                    "TrainConstantClock",
                    lambda s, a, increment=increment: [
                        Outcome(
                            {
                                "world.step_count": int(
                                    cast(int, s["world.step_count"])
                                )
                                + increment
                            }
                        )
                    ],
                    length=12,
                    source=f"step_count += {increment}",
                )
            )
            events.append({"slot": i, "proposal": "TrainConstantClock"})
        if slot.writes == frozenset({"player.action"}):
            action_map = {
                remap_slug_actions_to_balrog_actions(a): n["player.action"]
                for _, a, n in history
            }
            slot.triggers = None
            slot.candidates.append(
                Candidate(
                    "TrainActionRecording",
                    lambda s, a, mapping=action_map: [
                        Outcome(
                            {
                                "player.action": mapping.get(
                                    remap_slug_actions_to_balrog_actions(a),
                                    s["player.action"],
                                )
                            }
                        )
                    ],
                    length=20,
                    source=repr(action_map),
                )
            )
            events.append({"slot": i, "proposal": "TrainActionRecording"})
        if slot.writes == frozenset({"world.daylight"}):
            slot.candidates.append(
                Candidate(
                    "TrainPeriodicDaylight",
                    clock.predict,
                    length=40,
                    source=repr(clock),
                )
            )
            events.append({"slot": i, "proposal": "TrainPeriodicDaylight"})
        if slot.writes == frozenset(
            "player." + k for k in ("hunger", "thirst", "fatigue", "recover")
        ):
            for depth in (2, 4, 6):
                tree = infer_survival(history, depth)
                slot.candidates.append(
                    Candidate(
                        f"TrainSurvivalGuards{depth}",
                        tree.predict,
                        length=30 * depth,
                        source=repr(tree.root),
                    )
                )
                events.append(
                    {
                        "slot": i,
                        "proposal": f"TrainSurvivalGuards{depth}",
                        "root": tree.root,
                    }
                )
    return (
        posterior,
        adapter,
        history,
        {
            "clock": clock_report,
            "proposals": events,
            "birth_prototype_types": sorted(adapter.prototypes),
        },
    )
