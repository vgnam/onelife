"""Finite executable-law posteriors and Bayes-adaptive root-sampling search.

All predicates and effects read the *observed previous state*. Concurrent writes
compose in slot order (last writer wins). Exactness is relative to these declared
slots and finite outcome distributions; beam truncation is reported explicitly.
"""

import copy
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isclose, log, prod, sqrt
from typing import TypedDict

import numpy as np
from scipy.special import logsumexp

Value = Hashable
State = dict[str, Value]
MISSING = ("__belief_law_missing__",)


class ModelPartition(TypedDict):
    cone: list[str]
    pruned: list[int]
    fixed: list[int]
    sampled: list[int]
    epsilon_sum: float
    value_loss_bound: float
    policy_regret_bound: float


def equal(a: Value, b: Value) -> bool:
    if isinstance(a, float) and isinstance(b, (int, float)):
        return isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def matches(key: str, schemas: frozenset[str]) -> bool:
    parts = key.split(".")
    return any(
        len(parts) >= len(s.split("."))
        and all(a == "*" or a == b for a, b in zip(s.split("."), parts))
        for s in schemas
    )


def overlaps(left: frozenset[str], right: frozenset[str]) -> bool:
    return any(
        all(x == y or x == "*" or y == "*" for x, y in zip(a.split("."), b.split(".")))
        for a in left
        for b in right
    )


def changed(before: Mapping[str, Value], after: Mapping[str, Value]) -> State:
    return {
        k: after.get(k, MISSING)
        for k in before.keys() | after.keys()
        if not equal(before.get(k, MISSING), after.get(k, MISSING))
    }


@dataclass(frozen=True)
class Outcome:
    edits: Mapping[str, Value]
    probability: float = 1.0


@dataclass
class Candidate:
    name: str
    predict: Callable[[State, str], Sequence[Outcome]]
    length: int = 0
    source: str = ""
    learn_probabilities: bool = False
    kind: str = "program"


@dataclass
class Slot:
    name: str
    triggers: frozenset[str] | None
    writes: frozenset[str]
    reads: frozenset[str]
    candidates: list[Candidate]

    def compatible(self, action: str) -> bool:
        return self.triggers is None or action in self.triggers


def co_write_components(
    slots: Sequence[Slot], trigger_aware: bool = True
) -> list[tuple[int, ...]]:
    remaining = set(range(len(slots)))
    groups = []
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        group = []
        while pending:
            i = pending.pop()
            group.append(i)
            for j in sorted(remaining):
                left_trigger, right_trigger = slots[i].triggers, slots[j].triggers
                compatible = (
                    not trigger_aware
                    or left_trigger is None
                    or right_trigger is None
                    or bool(left_trigger & right_trigger)
                )
                if compatible and overlaps(slots[i].writes, slots[j].writes):
                    remaining.remove(j)
                    pending.append(j)
        groups.append(tuple(sorted(group)))
    return groups


@dataclass
class Component:
    slots: tuple[int, ...]
    assignments: list[tuple[int, ...]]
    log_weights: np.ndarray
    full_size: int
    counts: dict[tuple[tuple[int, ...], int, int], dict[int, float]] = field(
        default_factory=dict
    )

    @property
    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_weights - logsumexp(self.log_weights))


class CategoricalCatchAll:
    """Dirichlet-categorical joint edit predictor with a fixed context.

    The support is frozen before likelihood scoring. The Crafter adapter uses
    portable relative edits; this is a documented extension of the v2 table.
    The engine scores it prequentially, never with future counts.
    """

    def __init__(
        self,
        writes: frozenset[str],
        context: Callable[[State, str], Hashable],
        signature: Callable[[State, State], Hashable],
        restore: Callable[[State, Hashable], State],
        beta: float = 0.5,
    ):
        self.writes, self.context = writes, context
        self.signature, self.restore, self.beta = signature, restore, beta
        self.support: dict[Hashable, set[Hashable]] = defaultdict(lambda: {()})
        self.counts: dict[Hashable, Counter] = defaultdict(Counter)

    def observe_support(self, state: State, action: str, next_state: State) -> None:
        edits = {
            k: v
            for k, v in changed(state, next_state).items()
            if matches(k, self.writes)
        }
        self.support[self.context(state, action)].add(self.signature(state, edits))

    def predict(self, state: State, action: str) -> list[Outcome]:
        ctx = self.context(state, action)
        support = sorted(self.support.get(ctx, {()}), key=repr)
        counts = self.counts.get(ctx, Counter())
        denominator = sum(counts.values()) + self.beta * len(support)
        persistence = {
            v: value for v, value in state.items() if matches(v, self.writes)
        }
        return [
            Outcome(
                {**persistence, **self.restore(state, sig)},
                (counts[sig] + self.beta) / denominator,
            )
            for sig in support
        ]

    def update(self, state: State, action: str, next_state: State) -> None:
        edits = {
            k: v
            for k, v in changed(state, next_state).items()
            if matches(k, self.writes)
        }
        ctx, sig = self.context(state, action), self.signature(state, edits)
        # Online novelty expands the dictionary and requires full replay of the
        # affected posterior; fit() already freezes acquisition support first.
        if sig not in self.support[ctx]:
            self.support[ctx].add(sig)
        self.counts[ctx][sig] += 1


class FactoredLawPosterior:
    def __init__(
        self,
        slots: list[Slot],
        *,
        length_penalty: float = 0.01,
        beam: int = 512,
        outcome_budget: int = 4096,
        beta: float = 0.5,
        seed: int = 42,
    ):
        if beam < 1 or outcome_budget < 1 or beta <= 0:
            raise ValueError("Budgets and Dirichlet beta must be positive")
        self.slots, self.length_penalty = slots, length_penalty
        self.beam, self.outcome_budget, self.beta = beam, outcome_budget, beta
        self.rng = np.random.default_rng(seed)
        self.history: list[tuple[State, str, State]] = []
        self.catchalls: dict[int, CategoricalCatchAll] = {}
        self.approximations: Counter = Counter()
        self.evaluations = 0
        self.components: list[Component] = []
        self._reset_components()

    def _reset_components(self) -> None:
        self.components = []
        for indices in co_write_components(self.slots):
            size = prod(len(self.slots[i].candidates) for i in indices)
            # Incremental Cartesian beam avoids materializing an exponential
            # table before discovering that it exceeds the budget.
            entries: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
            for i in indices:
                entries = [
                    (z + (k,), weight - self.length_penalty * candidate.length)
                    for z, weight in entries
                    for k, candidate in enumerate(self.slots[i].candidates)
                ]
                entries.sort(key=lambda e: (-e[1], e[0]))
                entries = entries[: self.beam]
            # Retain all-null and all-catch-all escape assignments in a beam.
            for kind in ("null", "catchall"):
                selected = [
                    next(
                        (
                            k
                            for k, c in enumerate(self.slots[i].candidates)
                            if c.kind == kind
                        ),
                        None,
                    )
                    for i in indices
                ]
                if all(k is not None for k in selected):
                    z = tuple(k for k in selected if k is not None)
                    if z not in {e[0] for e in entries}:
                        weight = -self.length_penalty * sum(
                            self.slots[i].candidates[k].length
                            for i, k in zip(indices, z)
                        )
                        entries[-1 if kind == "null" or len(entries) == 1 else -2] = (
                            z,
                            weight,
                        )
            if size > self.beam:
                self.approximations["assignment_beam_components"] += 1
            self.components.append(
                Component(
                    indices,
                    [e[0] for e in entries],
                    np.array([e[1] for e in entries]),
                    size,
                )
            )

    def _events(
        self,
        component: Component,
        assignment: tuple[int, ...],
        state: State,
        action: str,
        cache: dict | None = None,
        parameter_sample: dict | None = None,
        observed: State | None = None,
    ) -> list[tuple[State, float, tuple]]:
        events = [({}, 1.0, ())]
        prepared = []
        for i, k in zip(component.slots, assignment):
            slot, candidate = self.slots[i], self.slots[i].candidates[k]
            if not slot.compatible(action):
                continue
            key = (i, k)
            if cache is not None and key in cache:
                outcomes = cache[key]
            else:
                # An open discrete catch-all can evaluate its unenumerated
                # branch at the observation instead of materializing infinity.
                conditional = getattr(candidate.predict, "__self__", None)
                analytic = observed is not None and hasattr(
                    conditional, "predict_observed"
                )
                outcomes = list(
                    conditional.predict_observed(state, action, observed)
                    if analytic
                    else candidate.predict(state, action)
                )
                if not outcomes or any(o.probability < 0 for o in outcomes):
                    raise ValueError(f"Invalid outcomes from {candidate.name}")
                total = sum(o.probability for o in outcomes)
                if total <= 0:
                    raise ValueError(f"Zero outcome mass from {candidate.name}")
                # Analytic outcomes contain only the mass relevant to the
                # observation, so renormalizing would change the likelihood.
                if not analytic:
                    outcomes = [
                        Outcome(o.edits, o.probability / total) for o in outcomes
                    ]
                for outcome in outcomes:
                    if any(not matches(v, slot.writes) for v in outcome.edits):
                        raise ValueError(
                            f"{candidate.name} writes outside slot {slot.name}"
                        )
                if cache is not None:
                    cache[key] = outcomes
            prepared.append((i, k, outcomes))
        future_writes = []
        future_guaranteed = []
        remaining = set()
        guaranteed = set()
        for _, _, outcomes in reversed(prepared):
            future_writes.append(set(remaining))
            future_guaranteed.append(set(guaranteed))
            remaining.update(v for o in outcomes for v in o.edits)
            guaranteed.update(set.intersection(*(set(o.edits) for o in outcomes)))
        future_writes.reverse()
        future_guaranteed.reverse()
        learn_latent = any(
            self.slots[i].candidates[k].learn_probabilities for i, k, _ in prepared
        )
        variables = set()
        if observed is not None:
            variable_key = ("variables", component.slots)
            writes = frozenset(
                s
                for i in component.slots
                if self.slots[i].compatible(action)
                for s in self.slots[i].writes
            )
            variables = (
                cache[variable_key]
                if cache is not None and variable_key in cache
                else {v for v in state.keys() | observed.keys() if matches(v, writes)}
            )
        for position, (i, k, outcomes) in enumerate(prepared):
            candidate = self.slots[i].candidates[k]
            probabilities = np.array([o.probability for o in outcomes])
            if (
                candidate.kind == "catchall"
                and parameter_sample is not None
                and i in self.catchalls
                and not getattr(self.catchalls[i], "integrated_probabilities", False)
            ):
                table = self.catchalls[i]
                ctx = table.context(state, action)
                support = sorted(table.support.get(ctx, {()}), key=repr)
                counts = table.counts.get(ctx, Counter())
                parameter_key = ("catchall", i, ctx)
                if parameter_key not in parameter_sample:
                    parameter_sample[parameter_key] = self.rng.dirichlet(
                        [table.beta + counts[sig] for sig in support]
                    )
                probabilities = parameter_sample[parameter_key]
            # No-op activation has no stochastic trial to count.
            informative = len(outcomes) > 1
            if candidate.learn_probabilities and informative:
                parameter_counts = component.counts.get((assignment, i, k), {})
                alpha = np.array(
                    [
                        self.beta + parameter_counts.get(j, 0.0)
                        for j in range(len(outcomes))
                    ]
                )
                if parameter_sample is None:
                    probabilities = alpha / alpha.sum()
                else:
                    parameter_key = (component.slots, assignment, i, k, len(outcomes))
                    if parameter_key not in parameter_sample:
                        parameter_sample[parameter_key] = self.rng.dirichlet(alpha)
                    probabilities = parameter_sample[parameter_key]
            combined = []
            finalized = variables - future_writes[position]
            for edits, p, ids in events:
                for j, outcome in enumerate(outcomes):
                    new_edits = {
                        v: value
                        for v, value in {**edits, **outcome.edits}.items()
                        if v not in future_guaranteed[position]
                    }
                    if observed is not None and any(
                        not equal(
                            new_edits.get(v, state.get(v, MISSING)),
                            observed.get(v, MISSING),
                        )
                        for v in finalized
                    ):
                        continue
                    combined.append(
                        (
                            new_edits,
                            p * probabilities[j],
                            ids
                            + (
                                ((i, k, j),)
                                if informative and candidate.learn_probabilities
                                else ()
                            ),
                        )
                    )
            if not learn_latent:
                merged = {}
                for edits, probability, _ in combined:
                    signature = tuple(sorted(edits.items()))
                    merged[signature] = merged.get(signature, 0.0) + probability
                combined = [
                    (dict(signature), probability, ())
                    for signature, probability in merged.items()
                ]
            if len(combined) > self.outcome_budget:
                self.approximations["outcome_beam_calls"] += 1
                combined.sort(key=lambda e: -e[1])
                combined = combined[: self.outcome_budget]
                if observed is None:
                    total = sum(e[1] for e in combined)
                    combined = [(edits, p / total, ids) for edits, p, ids in combined]
            events = combined
        return events

    def update(
        self, state: State, action: str, next_state: State, *, store: bool = True
    ) -> None:
        if store:
            self.history.append(
                (copy.deepcopy(state), action, copy.deepcopy(next_state))
            )
        for component in self.components:
            writes = frozenset(
                s
                for i in component.slots
                if self.slots[i].compatible(action)
                for s in self.slots[i].writes
            )
            if not writes:
                continue
            variables = {
                v for v in state.keys() | next_state.keys() if matches(v, writes)
            }
            cache = {("variables", component.slots): variables}
            likelihoods = []
            for z, old_weight in zip(component.assignments, component.log_weights):
                if not np.isfinite(old_weight):
                    likelihoods.append(-np.inf)
                    continue
                events = self._events(
                    component, z, state, action, cache, observed=next_state
                )
                possible = [
                    (p, ids)
                    for edits, p, ids in events
                    if all(
                        equal(
                            edits.get(v, state.get(v, MISSING)),
                            next_state.get(v, MISSING),
                        )
                        for v in variables | set(edits)
                    )
                ]
                likelihood = sum(p for p, _ in possible)
                likelihoods.append(log(likelihood) if likelihood > 0 else -np.inf)
                if likelihood > 0:
                    if len(possible) > 1 and any(
                        self.slots[i].candidates[k].learn_probabilities
                        for _, ids in possible
                        for i, k, _ in ids
                    ):
                        self.approximations["ambiguous_outcome_mean_updates"] += 1
                    for p, ids in possible:
                        for i, k, j in ids:
                            if self.slots[i].candidates[k].learn_probabilities:
                                counts = component.counts.setdefault(
                                    (z, i, k), defaultdict(float)
                                )
                                counts[j] += p / likelihood
                self.evaluations += 1
            weights = component.log_weights + np.array(likelihoods)
            if not np.isfinite(logsumexp(weights)):
                raise ValueError(
                    f"No assignment explains transition {len(self.history)} in {component.slots}; "
                    "expand the catch-all support and replay the library"
                )
            component.log_weights = weights - logsumexp(weights)
        for i, catchall in self.catchalls.items():
            if self.slots[i].compatible(action):
                catchall.update(state, action, next_state)

    def fit(
        self,
        history: Sequence[tuple[State, str, State]],
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.history = []
        for catchall in self.catchalls.values():
            catchall.counts.clear()
            catchall.support.clear()
        for state, action, nxt in history:
            for i, catchall in self.catchalls.items():
                if self.slots[i].compatible(action):
                    catchall.observe_support(state, action, nxt)
        self._reset_components()
        for step, (state, action, nxt) in enumerate(history, start=1):
            self.update(state, action, nxt)
            if progress is not None:
                progress(step, len(history))

    def observe(self, state: State, action: str, nxt: State) -> None:
        """Closed-loop update; dictionary expansion replays the stored history."""
        novel = False
        for i, catchall in self.catchalls.items():
            if not self.slots[i].compatible(action):
                continue
            edits = {
                k: v
                for k, v in changed(state, nxt).items()
                if matches(k, catchall.writes)
            }
            if catchall.signature(state, edits) not in catchall.support.get(
                catchall.context(state, action), {()}
            ):
                novel = True
                break
        if novel:
            self.approximations["empirical_catchall_support_expansions"] += 1
            self.fit([*self.history, (state, action, nxt)])
        else:
            self.update(state, action, nxt)

    def slot_mass(self, i: int, kind: str) -> float:
        for component in self.components:
            if i in component.slots:
                pos = component.slots.index(i)
                return float(
                    sum(
                        p
                        for z, p in zip(component.assignments, component.probabilities)
                        if self.slots[i].candidates[z[pos]].kind == kind
                    )
                )
        raise IndexError(i)

    def revise(
        self, i: int, proposals: Sequence[Candidate], *, floor: float = 1e-8
    ) -> None:
        """Add proposals with their length prior and replay every stored transition."""
        history = list(self.history)
        existing = {c.source or c.name for c in self.slots[i].candidates}
        self.slots[i].candidates.extend(
            c for c in proposals if (c.source or c.name) not in existing
        )
        self.fit(history)
        component = next(c for c in self.components if i in c.slots)
        pos = component.slots.index(i)
        marginals = np.zeros(len(self.slots[i].candidates))
        for z, p in zip(component.assignments, component.probabilities):
            marginals[z[pos]] += p
        keep = [
            c
            for k, c in enumerate(self.slots[i].candidates)
            if c.kind != "program" or marginals[k] >= floor
        ]
        if len(keep) != len(self.slots[i].candidates):
            self.slots[i].candidates = keep
            self.fit(history)

    def partition(
        self,
        goal_reads: frozenset[str],
        epsilon_budget: float = 0.05,
        horizon: int = 20,
        reward_max: float = 1.0,
    ) -> ModelPartition:
        if epsilon_budget < 0 or horizon < 1 or reward_max < 0:
            raise ValueError("Invalid reduction budget or reward range")
        cone = set(goal_reads)
        while True:
            expanded = cone | {
                read
                for slot in self.slots
                if overlaps(slot.writes, frozenset(cone))
                for read in slot.reads
            }
            if expanded == cone:
                break
            cone = expanded
        pruned, relevant = [], []
        for k, component in enumerate(self.components):
            if any(
                overlaps(self.slots[i].writes, frozenset(cone)) for i in component.slots
            ):
                relevant.append(k)
            else:
                pruned.append(k)
        errors = sorted(
            (1.0 - float(max(self.components[k].probabilities)), k) for k in relevant
        )
        fixed, total = [], 0.0
        for error, k in errors:
            if total + error <= epsilon_budget:
                fixed.append(k)
                total += error
        return {
            "cone": sorted(cone),
            "pruned": pruned,
            "fixed": fixed,
            "sampled": [k for k in relevant if k not in fixed],
            "epsilon_sum": total,
            "value_loss_bound": horizon * reward_max * total,
            "policy_regret_bound": 2 * horizon * reward_max * total,
        }

    def draw_model(
        self, partition: ModelPartition | None = None, *, map_only: bool = False
    ) -> "SampledModel":
        selected = {}
        for k, component in enumerate(self.components):
            if partition and k in partition["pruned"]:
                continue
            index = (
                int(np.argmax(component.probabilities))
                if map_only or (partition and k in partition["fixed"])
                else int(
                    self.rng.choice(
                        len(component.assignments), p=component.probabilities
                    )
                )
            )
            selected[k] = component.assignments[index]
        return SampledModel(self, selected)

    def sample_next_state(self, state: State, action: str) -> State:
        return self.draw_model().sample_next_state(state, action)

    def evaluate_log_probability(
        self, state: State, action: str, nxt: State, *, map_only: bool = False
    ) -> float:
        total, covered = 0.0, set()
        for component in self.components:
            writes = frozenset(
                s
                for i in component.slots
                if self.slots[i].compatible(action)
                for s in self.slots[i].writes
            )
            variables = {v for v in state.keys() | nxt.keys() if matches(v, writes)}
            covered.update(variables)
            if not variables:
                continue
            cache, terms = {("variables", component.slots): variables}, []
            assignments = (
                [(component.assignments[int(np.argmax(component.probabilities))], 1.0)]
                if map_only
                else zip(component.assignments, component.probabilities)
            )
            for z, pz in assignments:
                if pz == 0:
                    continue
                likelihood = sum(
                    p
                    for edits, p, _ in self._events(
                        component, z, state, action, cache, observed=nxt
                    )
                    if all(
                        equal(edits.get(v, state.get(v, MISSING)), nxt.get(v, MISSING))
                        for v in variables | set(edits)
                    )
                )
                if likelihood > 0:
                    terms.append(log(pz) + log(likelihood))
            if not terms:
                return -np.inf
            total += float(logsumexp(terms))
        if any(
            not equal(state.get(v, MISSING), nxt.get(v, MISSING))
            for v in (state.keys() | nxt.keys()) - covered
        ):
            return -np.inf
        return total

    def diagnostics(self) -> dict:
        return {
            "slots": len(self.slots),
            "components": len(self.components),
            "component_sizes": [c.full_size for c in self.components],
            "retained_sizes": [len(c.assignments) for c in self.components],
            "factored_assignments": sum(c.full_size for c in self.components),
            "unfactored_assignments": prod(len(s.candidates) for s in self.slots),
            "law_assignment_evaluations": self.evaluations,
            "approximations": dict(self.approximations),
            "slot_posteriors": [
                {
                    "name": s.name,
                    "triggers": sorted(s.triggers) if s.triggers else None,
                    "writes": sorted(s.writes),
                    "candidates": [c.name for c in s.candidates],
                    "catchall_mass": self.slot_mass(i, "catchall"),
                    "null_mass": self.slot_mass(i, "null"),
                }
                for i, s in enumerate(self.slots)
            ],
        }


class SampledModel:
    def __init__(
        self, posterior: FactoredLawPosterior, assignments: dict[int, tuple[int, ...]]
    ):
        self.posterior, self.assignments = posterior, assignments
        self.parameters: dict = {}

    def sample_next_state(self, state: State, action: str) -> State:
        nxt = dict(state)
        for k, z in self.assignments.items():
            component = self.posterior.components[k]
            events = self.posterior._events(
                component, z, state, action, parameter_sample=self.parameters
            )
            probabilities = np.array([p for _, p, _ in events])
            edits = events[
                int(
                    self.posterior.rng.choice(
                        len(events), p=probabilities / probabilities.sum()
                    )
                )
            ][0]
            for key, value in edits.items():
                if value == MISSING:
                    nxt.pop(key, None)
                else:
                    nxt[key] = value
        return nxt


@dataclass
class SearchNode:
    visits: int = 0
    action_visits: Counter = field(default_factory=Counter)
    values: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    children: dict = field(default_factory=dict)


class RootSamplingPlanner:
    """BAMCP-style history tree: one coherent model and phi per simulation.

    The history branches on observed state, so informative actions acquire value
    from their future action choices. No intrinsic information reward is used.
    """

    def __init__(
        self,
        posterior: FactoredLawPosterior,
        actions: Sequence[str],
        reward: Callable[[State, str, State], float],
        goal_reads: frozenset[str],
        *,
        horizon: int = 20,
        simulations: int = 200,
        epsilon_budget: float = 0.05,
        reward_max: float = 1.0,
        exploration: float = 1.4,
        mode: str = "belief",
    ):
        if (
            not actions
            or horizon < 1
            or simulations < 1
            or mode not in {"belief", "map", "psrl"}
        ):
            raise ValueError("Invalid planner configuration")
        self.posterior, self.actions, self.reward = posterior, list(actions), reward
        self.goal_reads, self.horizon, self.simulations = (
            goal_reads,
            horizon,
            simulations,
        )
        self.epsilon_budget, self.reward_max, self.exploration = (
            epsilon_budget,
            reward_max,
            exploration,
        )
        self.mode = mode

    def _simulate(
        self,
        node: SearchNode,
        state: State,
        model: SampledModel,
        depth: int,
        cone: frozenset[str],
    ) -> float:
        if depth == self.horizon:
            return 0.0
        untried = [a for a in self.actions if not node.action_visits[a]]
        if untried:
            action = untried[int(self.posterior.rng.integers(len(untried)))]
        else:
            action = max(
                self.actions,
                key=lambda a: (
                    node.values[a] / node.action_visits[a]
                    + self.exploration
                    * sqrt(log(max(1, node.visits)) / node.action_visits[a])
                ),
            )
        nxt = model.sample_next_state(state, action)
        immediate = self.reward(state, action, nxt)
        if not 0 <= immediate <= self.reward_max:
            raise ValueError("Reward violates the declared bound")
        observation = tuple(
            sorted((v, value) for v, value in nxt.items() if matches(v, cone))
        )
        child = node.children.setdefault((action, observation), SearchNode())
        value = immediate + self._simulate(child, nxt, model, depth + 1, cone)
        node.visits += 1
        node.action_visits[action] += 1
        node.values[action] += value
        return value

    def plan(self, state: State) -> tuple[str, dict]:
        partition: ModelPartition = self.posterior.partition(
            self.goal_reads, self.epsilon_budget, self.horizon, self.reward_max
        )
        if self.mode == "psrl":
            partition = {
                "cone": sorted(
                    set(state)
                    | set(self.goal_reads)
                    | {
                        s
                        for slot in self.posterior.slots
                        for s in slot.writes | slot.reads
                    }
                ),
                "pruned": [],
                "fixed": [],
                "sampled": list(range(len(self.posterior.components))),
                "epsilon_sum": 0.0,
                "value_loss_bound": 0.0,
                "policy_regret_bound": 0.0,
            }
        root = SearchNode()
        single = (
            self.posterior.draw_model(partition, map_only=self.mode == "map")
            if self.mode != "belief"
            else None
        )
        for _ in range(self.simulations):
            model = (
                single if single is not None else self.posterior.draw_model(partition)
            )
            self._simulate(root, state, model, 0, frozenset(partition["cone"]))
        action = max(
            self.actions,
            key=lambda a: (
                root.values[a] / root.action_visits[a]
                if root.action_visits[a]
                else -np.inf
            ),
        )
        return action, {
            "partition": partition,
            "simulations": self.simulations,
            "action_values": {
                a: root.values[a] / root.action_visits[a]
                for a in self.actions
                if root.action_visits[a]
            },
            "action_visits": dict(root.action_visits),
        }
