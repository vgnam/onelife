import copy
import math

import numpy as np
import pytest
from crafter_oo.functional_env import initial_state

from onelife.belief_law.core import Candidate, FactoredLawPosterior, Slot
from onelife.belief_law.crafter import CrafterAdapter
from onelife.belief_law.full_evaluation import (
    FullStateScorer,
    full_observations,
    universal_log_mass,
)
from onelife.belief_law.repair import (
    BackoffCatchAll,
    OpenScalarCatchAll,
    StructuralAdapter,
    infer_clock,
)


def table(cls=BackoffCatchAll, **kwargs):
    return cls(
        frozenset({"n"}),
        lambda s, a: a,
        lambda s, edits: tuple(edits.items()),
        lambda s, signature: dict(signature),
        **kwargs,
    )


def test_backoff_normalizes_and_uses_joint_support_for_unseen_context():
    catchall = table()
    catchall.observe_support({"n": 0}, "known", {"n": 7})
    catchall.update({"n": 0}, "known", {"n": 7})
    outcomes = catchall.predict({"n": 0}, "new")
    assert sum(o.probability for o in outcomes) == pytest.approx(1)
    assert any(o.edits["n"] == 7 for o in outcomes)
    assert catchall.counts.get("new") is None  # Prediction does not fit.


def test_open_escape_likelihood_is_not_renormalized_at_observation():
    catchall = OpenScalarCatchAll(
        frozenset({"world.step_count"}),
        lambda s, a: (),
        lambda s, e: tuple(e.items()),
        lambda s, sig: dict(sig),
        key="world.step_count",
    )
    posterior = FactoredLawPosterior(
        [
            Slot(
                "clock",
                None,
                frozenset({"world.step_count"}),
                frozenset(),
                [Candidate("escape", catchall.predict, kind="catchall")],
            )
        ]
    )
    posterior.catchalls[0] = catchall
    state = {"world.step_count": 10}
    expected = catchall.escape_mass / 3 / 2**7
    assert math.exp(
        posterior.evaluate_log_probability(state, "new", {"world.step_count": 17})
    ) == pytest.approx(expected)
    assert sum(
        catchall.base_probability(state, value) for value in range(-100, 120)
    ) == pytest.approx(1)
    np.random.seed(4)
    outcomes = catchall.predict(state, "new")
    assert sum(o.probability for o in outcomes) == pytest.approx(1)


def test_clock_induction_recovers_an_unseen_period_without_game_constants():
    period, phase = 240, 0.25
    history = []
    for step in range(2, 180):

        def state(t):
            return {
                "world.step_count": t,
                "world.daylight": float(
                    abs(np.cos(np.pi * ((t / period) % 1 + phase))) ** 2
                ),
            }

        history.append((state(step), "anything", state(step + 1)))
    law, report = infer_clock(history)
    assert report["max_train_error"] < 1e-9
    assert law.value(203) == pytest.approx(
        history[0][0]["world.daylight"] * 0
        + abs(np.cos(np.pi * ((203 / period) % 1 + phase))) ** 2,
        abs=1e-8,
    )


class DeterministicSampler:
    def sample_next_state(self, state, action):
        state.step_count += 1
        state.daylight = 0.321
        state.player.hunger += 1
        return state


def test_full_state_channel_distinguishes_previously_ignored_fields():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    correct = DeterministicSampler().sample_next_state(copy.deepcopy(state), "noop")
    scorer = FullStateScorer(DeterministicSampler(), samples=2)
    best = scorer.evaluate_log_probability(state, "noop", correct)
    fields = ["step_count", "daylight", "player.hunger", "player.action"]
    for field in fields:
        wrong = copy.deepcopy(correct)
        if field == "player.action":
            wrong.player.action = "unseen_action"
        elif field.startswith("player."):
            wrong.player.hunger += 100
        else:
            setattr(wrong, field, getattr(wrong, field) + 0.123)
        score = scorer.evaluate_log_probability(state, "noop", wrong)
        assert math.isfinite(score)
        assert score < best
    assert scorer.diagnostics()["nonfinite_scores"] == 0
    assert "objects" in full_observations(state)
    assert "chunks" in full_observations(state)


def test_birth_completion_and_id_counter_are_visible_in_emitted_state():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    original = CrafterAdapter(state)
    flat = original.encode(state)
    cow = {
        "entity.cow.9.name": "cow",
        "entity.cow.9.entity_id": 9,
        "entity.cow.9.position.x": 2,
        "entity.cow.9.position.y": 3,
        "entity.cow.9.health": 3,
        "entity.cow.9.removed": False,
    }
    adapter = StructuralAdapter(original, [(flat, "noop", {**flat, **cow})])
    partial = {**flat, "entity.cow.15.position.x": 4}
    decoded = adapter.decode(partial, state)
    new = next(o for o in decoded.objects if o.entity_id == 15)
    assert new.name == "cow" and new.position.x == 4 and new.health == 3
    assert decoded.entity_id_counter_state >= 16
    assert 15 in {i for chunk in decoded.chunks for i in chunk.objects}
    assert adapter.approximations["discarded_incomplete_entity_births"] == 0
    assert original.encode(state) == flat


def test_portable_birth_uses_current_counter_instead_of_train_id():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    original = CrafterAdapter(state)
    flat = original.encode(state)
    eid = flat["world.entity_id_counter_state"]
    edits = {
        f"entity.cow.{eid}.name": "cow",
        f"entity.cow.{eid}.entity_id": eid,
        f"entity.cow.{eid}.position.x": 2,
    }
    adapter = StructuralAdapter(original, [(flat, "noop", {**flat, **edits})])
    signature = adapter.signature(flat, edits)
    target = {**flat, "world.entity_id_counter_state": 71}
    restored = adapter.restore(target, signature)
    assert restored["entity.cow.71.entity_id"] == 71
    assert restored["entity.cow.71.position.x"] == 2
    # A different co-write component sees only the second birth's projection.
    partial = {f"entity.skeleton.{eid + 1}.entity_id": eid + 1}
    restored = adapter.restore(target, adapter.signature(flat, partial))
    assert restored["entity.skeleton.72.entity_id"] == 72


def test_full_channel_uses_exactly_the_old_equality_field_scope():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    correct = DeterministicSampler().sample_next_state(copy.deepcopy(state), "noop")
    equivalent = copy.deepcopy(correct)
    equivalent.view = (3, 3)
    equivalent.chunks = []
    assert equivalent == correct  # Old WorldState equality excludes both.
    scorer = FullStateScorer(DeterministicSampler(), samples=2)
    assert scorer.evaluate_log_probability(
        state, "noop", equivalent
    ) == scorer.evaluate_log_probability(state, "noop", correct)


def test_universal_noise_folds_equal_numeric_encodings():
    assert universal_log_mass(False) == universal_log_mass(0) == universal_log_mass(0.0)
    assert universal_log_mass(True) == universal_log_mass(1) == universal_log_mass(1.0)
