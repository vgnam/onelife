"""Crafter-OO adapter for BELIEF-LAW and legacy OneLife law programs.

Legacy programs expose marginal DiscreteDistribution objects rather than joint
outcomes. Their marginals are expanded independently; this assumption and any
outcome truncation are recorded. No true dynamics are used by this adapter.
"""

import ast
import copy
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import cast

import numpy as np
from crafter_oo.constants import actions
from crafter_oo.state_export import ChunkState, WorldState
from pydantic import BaseModel

from onelife.belief_law.core import (
    MISSING,
    Candidate,
    CategoricalCatchAll,
    FactoredLawPosterior,
    Outcome,
    Slot,
    State,
    changed,
)
from onelife.our_method.action_remapping import remap_slug_actions_to_balrog_actions
from onelife.our_method.core import LawProtocol
from onelife.poe_world.core import DiscreteDistribution


def freeze(value) -> Hashable:
    if isinstance(value, BaseModel):
        return tuple(
            (k, freeze(v)) for k, v in value.__dict__.items() if not k.startswith("_")
        )
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def schema(key: str) -> str:
    if key.startswith("tile."):
        return "tile"
    if key.startswith("entity."):
        parts = key.split(".")
        return ".".join(parts[:2] + ["*"] + parts[3:])
    return key


def coordinate(state: State, key: str) -> int:
    value = state[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"Non-numeric coordinate {key}")
    return int(value)


def declared_writes(source: str) -> set[str]:
    """Conservative AST write contracts, including unactivated programs.

    Resolve simple state aliases and typed entity loop variables. Unknown
    entity aliases retain the broad entity schema, never silently disappear.
    Runtime validation catches any unsupported Python alias idiom.
    """
    tree = ast.parse(source)
    aliases = {"current_state": "state", "state": "state"}
    entity_types = {"cow", "zombie", "skeleton", "arrow", "plant", "fence"}

    def path(node):
        if isinstance(node, ast.Name):
            if node.id in aliases:
                return aliases[node.id]
            kind = node.id.lower().rstrip("s")
            if kind in entity_types:
                return f"entity.{kind}.*"
            return ""
        if isinstance(node, ast.Attribute):
            parent = path(node.value)
            return parent + "." + node.attr if parent else ""
        return ""

    # Alias propagation until stable also resolves inv = player.inventory.
    for _ in range(5):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = path(node.value)
                if value:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            aliases[target.id] = value
            if isinstance(node, (ast.For, ast.comprehension)) and isinstance(
                node.target, ast.Name
            ):
                collection = node.iter
                kind = None
                if isinstance(collection, ast.Call):
                    for arg in collection.args:
                        if isinstance(arg, ast.Name) and arg.id.endswith("State"):
                            kind = arg.id[:-5].lower()
                if kind in entity_types:
                    aliases[node.target.id] = f"entity.{kind}.*"
    writes = set()
    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        for target in targets:
            if isinstance(target, ast.Attribute):
                value = path(target)
                if value.startswith("state.player."):
                    writes.add(value[6:])
                elif value.startswith("entity."):
                    writes.add(value)
                elif value.startswith("state.") and value.split(".")[1] in {
                    "daylight",
                    "step_count",
                    "entity_id_counter_state",
                    "chunks",
                }:
                    writes.add("world." + value[6:])
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_material"
        ):
            writes.add("tile")
    return writes


class CrafterAdapter:
    def __init__(self, template: WorldState, outcome_budget: int = 128):
        self.template = template
        self.outcome_budget = outcome_budget
        self.approximations: Counter = Counter()
        self.failures: Counter = Counter()
        self._decoded: dict[int, tuple[State, WorldState]] = {}

    def remember(self, flat: State, state: WorldState) -> None:
        if len(self._decoded) >= 512:
            self._decoded.clear()
        self._decoded[id(flat)] = (flat, state)

    def encode(self, state: WorldState) -> State:
        flat = {}

        def flatten(prefix, value):
            if isinstance(value, BaseModel):
                for k in type(value).model_fields:
                    flatten(prefix + "." + k, getattr(value, k))
            else:
                flat[prefix] = (
                    value if isinstance(value, DiscreteDistribution) else freeze(value)
                )

        flatten("player", state.player)
        for entity in state.objects:
            if entity.name != "player":
                flatten(f"entity.{entity.name}.{entity.entity_id}", entity)
        for x, column in enumerate(state.materials):
            for y, value in enumerate(column):
                flat[f"tile.{x}.{y}"] = value
        for key in (
            "daylight",
            "step_count",
            "entity_id_counter_state",
            "size",
            "view",
            "chunk_size",
        ):
            flat["world." + key] = freeze(getattr(state, key))
        flat["world.chunks"] = freeze(state.chunks)
        return flat

    def decode(self, state: State, reference: WorldState | None = None) -> WorldState:
        result = copy.deepcopy(reference or self.template)
        if reference is None:
            width, height = cast(tuple[int, int], state["world.size"])
            result.materials = [
                [cast(str | None, state.get(f"tile.{x}.{y}")) for y in range(height)]
                for x in range(width)
            ]
        for key, value in state.items():
            if key.startswith("player."):
                obj = result.player
                parts = key.split(".")[1:]
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif key.startswith("tile."):
                _, x, y = key.split(".")
                if 0 <= int(x) < len(result.materials) and 0 <= int(y) < len(
                    result.materials[int(x)]
                ):
                    result.materials[int(x)][int(y)] = cast(str | None, value)
            elif key.startswith("world.") and key != "world.chunks":
                setattr(result, key.split(".")[1], value)
        by_id = {
            (str(e.name), e.entity_id): copy.deepcopy(e)
            for e in result.objects
            if e.name != "player"
        }
        prefixes = {
            ".".join(k.split(".")[:3]) for k in state if k.startswith("entity.")
        }
        entities = []
        for prefix in sorted(prefixes):
            fields = {
                k[len(prefix) + 1 :]: v
                for k, v in state.items()
                if k.startswith(prefix + ".")
            }
            entity_id = int(prefix.split(".")[2])
            identity = (prefix.split(".")[1], entity_id)
            if identity in by_id:
                obj = by_id[identity]
                for path, value in fields.items():
                    target, parts = obj, path.split(".")
                    for part in parts[:-1]:
                        target = getattr(target, part)
                    setattr(target, parts[-1], value)
                entities.append(obj)
            else:
                required = {"name", "entity_id", "position.x", "position.y", "health"}
                required |= {
                    "zombie": {"cooldown"},
                    "skeleton": {"reload"},
                    "plant": {"grown", "ripe"},
                    "arrow": {"facing.x", "facing.y"},
                }.get(identity[0], set())
                if not required.issubset(fields):
                    self.approximations["discarded_incomplete_entity_births"] += 1
                    continue
                data = {}
                for path, value in fields.items():
                    target, parts = data, path.split(".")
                    for part in parts[:-1]:
                        target = target.setdefault(part, {})
                    target[parts[-1]] = value
                # Validation through the existing discriminated union also
                # handles newly-created entities without consulting the engine.
                prototype = result.model_dump()
                prototype["objects"] = [data]
                entities.append(WorldState.model_validate(prototype).objects[0])
        result.objects = [result.player] + entities
        if "world.chunks" in state:
            result.chunks = [
                ChunkState.model_validate(dict(chunk))
                for chunk in cast(
                    tuple[tuple[tuple[str, Hashable], ...], ...], state["world.chunks"]
                )
            ]
        return result

    def context(self, state: State, action: str) -> Hashable:
        x = coordinate(state, "player.position.x") + coordinate(
            state, "player.facing.x"
        )
        y = coordinate(state, "player.position.y") + coordinate(
            state, "player.facing.y"
        )
        material = state.get(f"tile.{x}.{y}", "outside")
        target = str(material)
        for key, value in state.items():
            if (
                key.startswith("entity.")
                and key.endswith(".position.x")
                and value == x
                and state.get(key[:-1] + "y") == y
            ):
                target = key.split(".")[1]
                break
        return remap_slug_actions_to_balrog_actions(action), target

    def _entity_map(self, state: State) -> dict[str, str]:
        by_type = defaultdict(set)
        for key in state:
            if key.startswith("entity."):
                _, kind, eid, *_ = key.split(".")
                by_type[kind].add(int(eid))
        return {
            f"entity.{kind}.{eid}": f"entity.{kind}.@{rank}"
            for kind, ids in by_type.items()
            for rank, eid in enumerate(sorted(ids))
        }

    def signature(self, state: State, edits: State) -> Hashable:
        entity_map, result = self._entity_map(state), []
        for key, value in sorted(edits.items()):
            portable = key
            if key.startswith("tile."):
                _, x, y = key.split(".")
                portable = f"tile.@{int(x) - coordinate(state, 'player.position.x')}.@{int(y) - coordinate(state, 'player.position.y')}"
            elif key.startswith("entity."):
                prefix = ".".join(key.split(".")[:3])
                portable = entity_map.get(prefix, prefix) + key[len(prefix) :]
            old = state.get(key, MISSING)
            if (
                isinstance(value, (int, float))
                and isinstance(old, (int, float))
                and not isinstance(value, bool)
            ):
                result.append((portable, "delta", round(value - old, 10)))
            else:
                result.append((portable, "set", value))
        return tuple(result)

    def restore(self, state: State, signature: Hashable) -> State:
        inverse = {v: k for k, v in self._entity_map(state).items()}
        result = {}
        for key, mode, value in cast(tuple[tuple[str, str, Hashable], ...], signature):
            if key.startswith("tile.@"):
                _, x, y = key.split(".")
                key = f"tile.{int(x[1:]) + coordinate(state, 'player.position.x')}.{int(y[1:]) + coordinate(state, 'player.position.y')}"
                if key not in state:
                    continue
            elif key.startswith("entity."):
                prefix = ".".join(key.split(".")[:3])
                if "@" in prefix and prefix not in inverse:
                    continue
                key = inverse.get(prefix, prefix) + key[len(prefix) :]
            if mode == "delta":
                old = state.get(key, 0)
                if not isinstance(old, (int, float)) or not isinstance(
                    value, (int, float)
                ):
                    raise TypeError(f"Non-numeric relative edit {key}")
                result[key] = old + value
            else:
                result[key] = value
        return result

    def predictor(self, law: LawProtocol) -> Callable[[State, str], list[Outcome]]:
        def predict(state: State, action: str) -> list[Outcome]:
            # Use the raw object to count errors instead of silently swallowing
            # them in the legacy wrapper. It receives the same action remap.
            raw = getattr(law, "law", law)
            remembered = self._decoded.get(id(state))
            decoded = (
                copy.deepcopy(remembered[1])
                if remembered and remembered[0] is state
                else self.decode(state)
            )
            remapped = remap_slug_actions_to_balrog_actions(action)
            try:
                if not raw.precondition(decoded, remapped):
                    return [Outcome({})]
                raw.effect(decoded, remapped)
                after = self.encode(decoded)
            except (
                AttributeError,
                TypeError,
                ValueError,
                IndexError,
                KeyError,
            ) as error:
                self.failures[f"{law.__name__}:{type(error).__name__}"] += 1
                return [Outcome({})]
            edits = changed(state, after)
            distributions = [
                (k, v) for k, v in edits.items() if isinstance(v, DiscreteDistribution)
            ]
            deterministic = {
                k: v
                for k, v in edits.items()
                if not isinstance(v, DiscreteDistribution)
            }
            if not distributions:
                return [Outcome(deterministic)]
            if len(distributions) > 1:
                self.approximations["legacy_independent_marginal_calls"] += 1
            # Incremental highest-probability Cartesian beam.
            outcomes = [Outcome(deterministic)]
            for key, distribution in distributions:
                probabilities = np.exp(distribution.log_probs)
                expanded = [
                    Outcome({**o.edits, key: freeze(v)}, o.probability * float(p))
                    for o in outcomes
                    for v, p in zip(distribution.support, probabilities)
                    if p > 0
                ]
                if len(expanded) > self.outcome_budget:
                    self.approximations["legacy_outcome_beam_calls"] += 1
                    expanded.sort(key=lambda o: -o.probability)
                    expanded = expanded[: self.outcome_budget]
                total = sum(o.probability for o in expanded)
                outcomes = [Outcome(o.edits, o.probability / total) for o in expanded]
            return outcomes

        return predict


def build_library(
    adapter: CrafterAdapter,
    laws: Sequence[LawProtocol],
    history: Sequence[tuple[State, str, State]],
    *,
    beam: int = 512,
    length_penalty: float = 0.01,
    seed: int = 42,
) -> FactoredLawPosterior:
    """Use the *same acquired programs* as OneLife, without its fitted weights.

    Write sets are discovered on acquisition states and all action types on a
    small acquisition probe. Runtime write contracts are enforced by the core.
    Reads are conservative full-state sets for opaque legacy Python programs.
    This deliberately sacrifices Tier 0 pruning rather than claiming an unsafe
    dependency analysis is lossless.
    """
    groups = defaultdict(list)
    all_reads = frozenset({"player", "entity", "tile", "world"})
    probes = list(history)
    if history:
        probes += [(s, a, s) for s, _, _ in history[:3] for a in actions]
    for law in laws:
        predictor, writes, active_actions = (
            adapter.predictor(law),
            declared_writes(law.__source_code__),
            set(),
        )
        for state, action, _ in probes:
            outcomes = predictor(state, action)
            touched = {schema(k) for o in outcomes for k in o.edits}
            if touched:
                writes.update(touched)
                active_actions.add(action)
        if not writes:
            # Preserve unactivated candidates in a conservative slot instead of
            # pretending that unobserved mechanisms cannot exist.
            writes = {"player", "entity", "tile", "world"}
            triggers = None
        else:
            # Passive is the safe choice unless the Python source explicitly
            # mentions action constants. Probe activation expands this set.
            constants = {
                n.value
                for n in ast.walk(ast.parse(law.__source_code__))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
            named = {
                a
                for a in actions
                if a in constants
                or remap_slug_actions_to_balrog_actions(a) in constants
            }
            triggers = frozenset(named | active_actions) if named else None
        source = law.__source_code__
        candidate = Candidate(
            law.__name__, predictor, len(list(ast.walk(ast.parse(source)))), source
        )
        groups[(triggers, frozenset(writes))].append(candidate)
    observed_schemas = {schema(v) for s, _, nxt in history for v in changed(s, nxt)}
    covered = {s for _, writes in groups for s in writes}
    for item in sorted(observed_schemas):
        if not any(item == c or item.startswith(c + ".") for c in covered):
            groups[(None, frozenset({item}))] = []
    slots, catchalls = [], {}
    for index, ((triggers, writes), candidates) in enumerate(groups.items()):
        catchall = CategoricalCatchAll(
            writes, adapter.context, adapter.signature, adapter.restore
        )
        candidates = [
            Candidate("null", lambda s, a: [Outcome({})], kind="null"),
            *candidates,
            Candidate("catchall", catchall.predict, length=500, kind="catchall"),
        ]
        slots.append(Slot(f"slot_{index}", triggers, writes, all_reads, candidates))
        catchalls[index] = catchall
    posterior = FactoredLawPosterior(
        slots, beam=beam, length_penalty=length_penalty, seed=seed
    )
    posterior.catchalls = catchalls
    return posterior


class CrafterBeliefWorldModel:
    def __init__(
        self,
        posterior: FactoredLawPosterior,
        adapter: CrafterAdapter,
        *,
        map_only: bool = False,
    ):
        self.posterior, self.adapter, self.map_only = posterior, adapter, map_only

    def sample_next_state(self, current_state: WorldState, action: str) -> WorldState:
        flat = self.adapter.encode(current_state)
        self.adapter.remember(flat, current_state)
        nxt = self.posterior.draw_model(map_only=self.map_only).sample_next_state(
            flat, action
        )
        return self.adapter.decode(nxt, current_state)

    def evaluate_log_probability(
        self, state: WorldState, action: str, next_state: WorldState
    ) -> float:
        before, after = self.adapter.encode(state), self.adapter.encode(next_state)
        self.adapter.remember(before, state)
        # Legacy evaluator ignores some observable state fields. The adapter
        # includes all symbolic fields, so its likelihood is stricter.
        return self.posterior.evaluate_log_probability(
            before, action, after, map_only=self.map_only
        )
