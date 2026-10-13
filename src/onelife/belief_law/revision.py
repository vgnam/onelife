"""Localized, catch-all-triggered LLM proposals with a replayed length prior."""

import ast
import json
import os
from hashlib import sha256
from math import log
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from openai import OpenAI
from pydantic import BaseModel

from onelife.belief_law.core import (
    Candidate,
    CategoricalCatchAll,
    Slot,
    changed,
    matches,
)
from onelife.belief_law.crafter import CrafterAdapter, schema
from onelife.our_method.crafter.serde import _materialize_law


class ProposedLaw(BaseModel):
    law_code: str


class Proposal(BaseModel):
    candidates: list[ProposedLaw]


class LibraryReviser:
    def __init__(
        self,
        adapter: CrafterAdapter,
        cache_dir: Path,
        *,
        threshold: float = 0.5,
        max_calls: int = 4,
        window: int = 50,
    ):
        self.adapter, self.cache_dir = adapter, cache_dir
        self.threshold, self.max_calls, self.window = threshold, max_calls, window
        self.calls = 0
        self.network_calls = 0
        self.last_call: dict[str, int] = {}
        self.events: list[dict] = []
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _evidence(self, posterior, index):
        slot = posterior.slots[index]
        evidence = []
        for state, action, nxt in posterior.history:
            if not slot.compatible(action):
                continue
            edits = {
                k: v for k, v in changed(state, nxt).items() if matches(k, slot.writes)
            }
            likelihoods = []
            for candidate in slot.candidates:
                if candidate.kind != "program":
                    continue
                outcomes = candidate.predict(state, action)
                variables = {
                    v for v in state.keys() | nxt.keys() if matches(v, slot.writes)
                }
                likelihoods.append(
                    sum(
                        o.probability
                        for o in outcomes
                        if all(
                            o.edits.get(v, state.get(v)) == nxt.get(v)
                            for v in variables
                        )
                    )
                )
            surprise = -log(max([1e-100, *likelihoods]))
            # Compact fully observed context, enough to write portable code;
            # no held-out benchmark transitions enter the proposal prompt.
            player = {k: v for k, v in state.items() if k.startswith("player.")}
            entities = {k: v for k, v in state.items() if k.startswith("entity.")}
            x, y = int(state["player.position.x"]), int(state["player.position.y"])
            local_tiles = {
                k: v
                for k, v in state.items()
                if k.startswith("tile.")
                and abs(int(k.split(".")[1]) - x) <= 2
                and abs(int(k.split(".")[2]) - y) <= 2
            }
            evidence.append(
                (
                    surprise,
                    {
                        "action": action,
                        "player": player,
                        "entities": entities,
                        "tiles": local_tiles,
                        "world": {
                            k: v
                            for k, v in state.items()
                            if k.startswith("world.") and k != "world.chunks"
                        },
                        "observed_edits": edits,
                    },
                )
            )
        evidence.sort(key=lambda e: -e[0])
        unique = {}
        for _, item in evidence:
            signature = json.dumps(item["observed_edits"], sort_keys=True)
            unique.setdefault(signature, item)
        return list(unique.values())[:5]

    def run(self, posterior, *, preview: bool = False) -> int:
        touched = 0
        # Snapshot indices: new slots created by proposals are processed on a
        # subsequent window, under the same per-episode call cap.
        indices = sorted(
            range(len(posterior.slots)),
            key=lambda i: -posterior.slot_mass(i, "catchall"),
        )
        for index in indices:
            mass = posterior.slot_mass(index, "catchall")
            slot = posterior.slots[index]
            step = len(posterior.history)
            if mass <= self.threshold or self.calls >= self.max_calls:
                continue
            if step - self.last_call.get(slot.name, -self.window) < self.window:
                continue
            self.calls += 1
            self.last_call[slot.name] = step
            evidence = self._evidence(posterior, index)
            program_sources = [c.source for c in slot.candidates if c.kind == "program"]
            prompt = (
                "Infer 2 distinct plausible executable alternatives for ONE Crafter-OO mechanism. "
                'Use only the supplied acquisition data. Return JSON {"candidates": [{"law_code": "class ..."}, ...]}. '
                "Each class has precondition(current_state: WorldState, action: str)->bool and "
                "effect(current_state: WorldState, action: str)->None. Effects mutate a copy. "
                "Use DiscreteDistribution(support=[...], logscores=[...]) for uncertain numeric values; "
                "direct assignments for deterministic strings/bools. No imports, no hidden engine access, "
                "no memorize-by-coordinate rules. action uses labels Do, Sleep, Move East, Make Wood Pickaxe etc. "
                "WorldState.player has inventory, achievements, position(x,y), facing(x,y), health, sleeping, "
                "hunger, thirst, fatigue, recover, last_health, action. objects contains player and entities "
                "with name, entity_id, position, health; zombie.cooldown, skeleton.reload, plant.grown/ripe. "
                "state.get_target_tile()->(material, entity), state.set_material(x,y,str), "
                "state.get_object_of_type_in_update_range(CowState etc) are available. "
                "world daylight, step_count, entity_id_counter_state, chunks also exist. "
                "Propose alternatives for the declared writes; if an effect necessarily writes more fields, "
                "include them so the caller can create a new slot. Do not return confidence.\n"
                + json.dumps(
                    {
                        "slot": slot.name,
                        "writes": sorted(slot.writes),
                        "triggers": sorted(slot.triggers) if slot.triggers else None,
                        "existing_programs": program_sources,
                        "evidence": evidence,
                    },
                    ensure_ascii=False,
                )
            )
            fingerprint = sha256(prompt.encode()).hexdigest()[:16]
            path = self.cache_dir / f"{slot.name}_{fingerprint}.json"
            start = perf_counter()
            if preview:
                path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
                self.events.append(
                    {
                        "slot": slot.name,
                        "trigger_mass": mass,
                        "pending": True,
                        "prompt_path": str(path.with_suffix(".prompt.txt").resolve()),
                        "model": os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                        "destination_host": urlsplit(
                            os.environ.get(
                                "OPENAI_BASE_URL", "https://api.openai.com/v1"
                            )
                        ).netloc,
                    }
                )
                continue
            if path.exists():
                response = json.loads(path.read_text(encoding="utf-8"))
                cached = True
            else:
                client = OpenAI(max_retries=0)
                self.network_calls += 1
                result = client.chat.completions.create(
                    model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16000,
                    response_format={"type": "json_object"},
                    timeout=180,
                )
                content = result.choices[0].message.content
                if not content:
                    raise ValueError("Empty law proposal response")
                response = {
                    "proposal": json.loads(content),
                    "usage": result.usage.model_dump() if result.usage else None,
                    "model": result.model,
                    "prompt_sha256": sha256(prompt.encode()).hexdigest(),
                }
                path.write_text(json.dumps(response, indent=2), encoding="utf-8")
                path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
                cached = False
            proposals, expanded = [], []
            rejected = []
            proposal = Proposal.model_validate(response["proposal"])
            for item in proposal.candidates:
                source = item.law_code
                try:
                    law = _materialize_law(source)
                    predictor = self.adapter.predictor(law)
                    writes = {
                        schema(k)
                        for s, a, _ in posterior.history
                        for outcome in predictor(s, a)
                        for k in outcome.edits
                    }
                    candidate = Candidate(
                        law.__name__,
                        predictor,
                        len(list(ast.walk(ast.parse(source)))),
                        source,
                    )
                    if not writes:
                        rejected.append(
                            {
                                "name": law.__name__,
                                "reason": "No observed effect; cannot verify write contract",
                            }
                        )
                    elif all(matches(w, slot.writes) for w in writes):
                        proposals.append(candidate)
                    else:
                        expanded.append((candidate, frozenset(writes)))
                except (ValueError, SyntaxError, TypeError, KeyError) as error:
                    rejected.append({"reason": str(error)})
            if proposals:
                posterior.revise(index, proposals)
                touched += 1
            for candidate, writes in expanded:
                catchall = CategoricalCatchAll(
                    writes,
                    self.adapter.context,
                    self.adapter.signature,
                    self.adapter.restore,
                )
                new = Slot(
                    f"revision_{len(posterior.slots)}",
                    None,
                    writes,
                    frozenset({"player", "entity", "tile", "world"}),
                    [
                        posterior.slots[index].candidates[0],
                        candidate,
                        Candidate(
                            "catchall", catchall.predict, length=500, kind="catchall"
                        ),
                    ],
                )
                posterior.catchalls[len(posterior.slots)] = catchall
                posterior.slots.append(new)
                posterior.fit(list(posterior.history))
                touched += 1
            self.events.append(
                {
                    "slot": slot.name,
                    "trigger_mass": mass,
                    "cached": cached,
                    "accepted": [c.name for c in proposals],
                    "new_slots": [c.name for c, _ in expanded],
                    "rejected": rejected,
                    "seconds": perf_counter() - start,
                    "usage": response.get("usage"),
                }
            )
            (self.cache_dir / "revision_events.json").write_text(
                json.dumps(self.events, indent=2), encoding="utf-8"
            )
        (self.cache_dir / "revision_events.json").write_text(
            json.dumps(self.events, indent=2), encoding="utf-8"
        )
        return touched
