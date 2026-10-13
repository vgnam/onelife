"""Falsifiable requirements from the supplied BELIEF-LAW v2 specification."""

from itertools import product
from math import log
from typing import cast

import numpy as np
import pytest

from onelife.belief_law.core import (
    Candidate,
    CategoricalCatchAll,
    FactoredLawPosterior,
    Outcome,
    RootSamplingPlanner,
    Slot,
    State,
    Value,
    co_write_components,
)


def constant(name, value, key="x"):
    return Candidate(name, lambda s, a: [Outcome({key: value})])


def slot(name, trigger, writes, candidates, reads=()):
    return Slot(
        name,
        None if trigger is None else frozenset(trigger),
        frozenset(writes),
        frozenset(reads),
        candidates,
    )


def test_trigger_aware_factorization_does_not_merge_exclusive_actions():
    slots = [
        slot("chop", ["chop"], ["wood"], [constant("c", 1, "wood")]),
        slot("craft", ["craft"], ["wood"], [constant("c", 0, "wood")]),
        slot("cow", None, ["cow"], [constant("c", 0, "cow")]),
    ]
    assert co_write_components(slots) == [(0,), (1,), (2,)]
    assert co_write_components(slots, trigger_aware=False) == [(0, 1), (2,)]
    slots.append(slot("decay", None, ["wood"], [constant("c", 0, "wood")]))
    assert co_write_components(slots) == [(0, 1, 3), (2,)]


def test_factored_posterior_matches_full_joint_enumeration():
    slots = [
        slot(
            "x",
            ["go"],
            ["x"],
            [
                Candidate(
                    "a", lambda s, a: [Outcome({"x": 1}, 0.8), Outcome({"x": 0}, 0.2)]
                ),
                Candidate(
                    "b", lambda s, a: [Outcome({"x": 1}, 0.3), Outcome({"x": 0}, 0.7)]
                ),
            ],
        ),
        slot(
            "y",
            ["go"],
            ["y"],
            [
                Candidate(
                    "c", lambda s, a: [Outcome({"y": 1}, 0.6), Outcome({"y": 0}, 0.4)]
                ),
                Candidate(
                    "d", lambda s, a: [Outcome({"y": 1}, 0.1), Outcome({"y": 0}, 0.9)]
                ),
            ],
        ),
    ]
    posterior = FactoredLawPosterior(slots)
    posterior.fit([({"x": 0, "y": 0}, "go", {"x": 1, "y": 0})])
    expected = np.array([0.8 * 0.4, 0.8 * 0.9, 0.3 * 0.4, 0.3 * 0.9])
    expected /= expected.sum()
    actual = np.array(
        [
            posterior.components[0].probabilities[i]
            * posterior.components[1].probabilities[j]
            for i, j in product(range(2), repeat=2)
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_co_write_likelihood_uses_known_composition_order():
    slots = [
        slot("first", None, ["x"], [constant("zero", 0), constant("one", 1)]),
        slot("second", None, ["x"], [constant("two", 2), constant("three", 3)]),
    ]
    p = FactoredLawPosterior(slots)
    p.update({"x": 0}, "go", {"x": 2})
    assert p.components[0].probabilities.tolist() == [0.5, 0, 0.5, 0]


def test_joint_outcomes_preserve_correlated_effects():
    candidate = Candidate(
        "chop",
        lambda s, a: [
            Outcome({"tree": 0, "wood": 1}, 0.5),
            Outcome({"tree": 1, "wood": 0}, 0.5),
        ],
    )
    p = FactoredLawPosterior([slot("chop", None, ["tree", "wood"], [candidate])])
    state: State = {"tree": 1, "wood": 0}
    assert p.evaluate_log_probability(state, "do", {"tree": 0, "wood": 0}) == -np.inf
    outcomes = {
        tuple(sorted(p.sample_next_state(state, "do").items())) for _ in range(100)
    }
    assert len(outcomes) == 2


def test_read_dependency_does_not_couple_fully_observed_laws():
    p = FactoredLawPosterior(
        [
            slot("x", None, ["x"], [constant("a", 1)], ["y"]),
            slot("y", None, ["y"], [constant("b", 1)], ["x"]),
        ]
    )
    assert len(p.components) == 2


def test_null_persistence_and_unowned_variable_persistence():
    p = FactoredLawPosterior(
        [
            slot(
                "x",
                None,
                ["x"],
                [Candidate("null", lambda s, a: [Outcome({})], kind="null")],
            )
        ]
    )
    state: State = {"x": 2, "y": 3}
    assert p.sample_next_state(state, "go") == state
    assert p.evaluate_log_probability(state, "go", {"x": 2, "y": 4}) == -np.inf


def test_reduction_closes_reads_and_respects_total_epsilon_budget():
    p = FactoredLawPosterior(
        [
            slot(
                "goal",
                None,
                ["goal"],
                [constant("a", 1, "goal"), constant("b", 0, "goal")],
                ["x"],
            ),
            slot("x", None, ["x"], [constant("a", 1), constant("b", 0)]),
            slot("cow", None, ["cow"], [constant("a", 1, "cow")]),
        ]
    )
    p.components[0].log_weights = np.log([0.99, 0.01])
    p.components[1].log_weights = np.log([0.98, 0.02])
    partition = p.partition(frozenset({"goal"}), 0.025, 10)
    assert partition["pruned"] == [2]
    assert partition["fixed"] == [0]
    assert partition["sampled"] == [1]
    assert partition["policy_regret_bound"] == pytest.approx(0.2)


def test_dirichlet_identifiable_outcome_posterior_predictive():
    candidate = Candidate(
        "coin",
        lambda s, a: [Outcome({"x": 0}, 0.5), Outcome({"x": 1}, 0.5)],
        learn_probabilities=True,
    )
    p = FactoredLawPosterior([slot("coin", None, ["x"], [candidate])], beta=1)
    p.fit([({"x": 0}, "flip", {"x": 1})] * 3)
    assert p.evaluate_log_probability({"x": 0}, "flip", {"x": 1}) == pytest.approx(
        log(4 / 5)
    )


def test_root_sample_keeps_law_identity_for_entire_rollout():
    p = FactoredLawPosterior(
        [slot("x", None, ["x"], [constant("a", 0), constant("b", 1)])]
    )
    model = p.draw_model()
    values = [model.sample_next_state({"x": 0}, "go")["x"] for _ in range(50)]
    assert len(set(values)) == 1


def test_beam_is_bounded_and_explicitly_logged():
    slots = [
        slot(str(i), None, ["x"], [constant(str(k), k) for k in range(4)])
        for i in range(8)
    ]
    p = FactoredLawPosterior(slots, beam=10)
    assert len(p.components[0].assignments) == 10
    assert p.diagnostics()["component_sizes"] == [4**8]
    assert p.approximations["assignment_beam_components"] == 1


def test_revision_rescores_proposals_on_full_history_not_confidence():
    p = FactoredLawPosterior([slot("x", None, ["x"], [constant("old", 1)])])
    p.fit([({"x": 0}, "go", {"x": 1})] * 4)
    p.revise(0, [constant("wrong", 0), constant("correct", 1)])
    assert "wrong" not in [c.name for c in p.slots[0].candidates]
    assert len(p.history) == 4


def test_catchall_counts_are_prequential_and_context_specific():
    c = CategoricalCatchAll(
        frozenset({"x"}),
        lambda s, a: a,
        lambda s, edits: tuple(sorted(edits.items())),
        lambda s, sig: dict(cast(tuple[tuple[str, Value], ...], sig)),
        beta=1,
    )
    c.observe_support({"x": 0}, "go", {"x": 1})
    assert [o.probability for o in c.predict({"x": 0}, "go")] == [0.5, 0.5]
    c.update({"x": 0}, "go", {"x": 1})
    assert sorted(o.probability for o in c.predict({"x": 0}, "go")) == pytest.approx(
        [1 / 3, 2 / 3]
    )
    assert c.predict({"x": 0}, "other") == [Outcome({"x": 0}, 1)]


def test_history_tree_values_information_without_information_bonus():
    def rule(answer):
        def predict(s, action):
            if s["done"]:
                return [Outcome({})]
            if action == "inspect":
                return [Outcome({"hint": answer})]
            if action.startswith("choose"):
                return [Outcome({"goal": int(action == f"choose{answer}"), "done": 1})]
            return [Outcome({})]

        return Candidate(str(answer), predict)

    p = FactoredLawPosterior(
        [slot("hidden_law", None, ["hint", "goal", "done"], [rule(0), rule(1)])],
        seed=10,
    )
    reward = lambda s, a, n: float(
        not s["done"] and a.startswith("choose") and n["goal"] == 1
    )
    planner = RootSamplingPlanner(
        p,
        ["inspect", "choose0", "choose1"],
        reward,
        frozenset({"hint", "goal"}),
        horizon=2,
        simulations=3000,
        epsilon_budget=0,
    )
    action, details = planner.plan({"hint": -1, "goal": 0, "done": 0})
    assert max(details["action_values"].values()) >= 0.9
    assert action == "inspect"


def test_out_of_slot_writes_are_rejected():
    p = FactoredLawPosterior([slot("x", None, ["x"], [constant("bad", 1, "y")])])
    with pytest.raises(ValueError, match="outside slot"):
        p.sample_next_state({"x": 0, "y": 0}, "go")


def test_overwritten_stochastic_outcomes_marginalize_without_losing_mass():
    early = Candidate(
        "early", lambda s, a: [Outcome({"x": 0}, 0.9), Outcome({"x": 1}, 0.1)]
    )
    late = Candidate(
        "late", lambda s, a: [Outcome({"x": 2}, 0.3), Outcome({"x": 3}, 0.7)]
    )
    p = FactoredLawPosterior(
        [slot("early", None, ["x"], [early]), slot("late", None, ["x"], [late])]
    )
    assert p.evaluate_log_probability({"x": 0}, "go", {"x": 2}) == pytest.approx(
        log(0.3)
    )
    assert p.evaluate_log_probability({"x": 0}, "go", {"x": 3}) == pytest.approx(
        log(0.7)
    )


def test_online_catchall_novelty_replays_history_before_updating_belief():
    catchall = CategoricalCatchAll(
        frozenset({"x"}),
        lambda s, a: a,
        lambda s, edits: tuple(sorted(edits.items())),
        lambda s, sig: dict(cast(tuple[tuple[str, Value], ...], sig)),
    )
    p = FactoredLawPosterior(
        [
            slot(
                "x",
                None,
                ["x"],
                [Candidate("catchall", catchall.predict, kind="catchall")],
            )
        ]
    )
    p.catchalls[0] = catchall
    p.fit([({"x": 0}, "go", {"x": 1})])
    p.observe({"x": 1}, "go", {"x": 2})
    assert len(p.history) == 2
    assert p.evaluate_log_probability({"x": 0}, "go", {"x": 2}) > -np.inf
    assert p.approximations["empirical_catchall_support_expansions"] == 1


def test_psrl_uses_one_full_model_per_decision_at_same_simulation_budget(monkeypatch):
    p = FactoredLawPosterior(
        [
            slot("x", None, ["x"], [constant("a", 0), constant("b", 1)]),
            slot("irrelevant", None, ["cow"], [constant("a", 0, "cow")]),
        ]
    )
    draws = []
    original = p.draw_model

    def draw(*args, **kwargs):
        model = original(*args, **kwargs)
        draws.append(model)
        return model

    monkeypatch.setattr(p, "draw_model", draw)
    planner = RootSamplingPlanner(
        p,
        ["go"],
        lambda s, a, n: 0.0,
        frozenset({"x"}),
        horizon=2,
        simulations=20,
        epsilon_budget=1,
        mode="psrl",
    )
    _, details = planner.plan({"x": 0, "cow": 0})
    assert len(draws) == 1
    assert details["partition"]["pruned"] == []
    assert details["partition"]["fixed"] == []
    assert details["simulations"] == 20
