import copy
from collections import defaultdict
from types import SimpleNamespace

import pytest
from crafter_oo.functional_env import initial_state

from onelife.belief_law.core import (
    Candidate,
    CategoricalCatchAll,
    FactoredLawPosterior,
    Outcome,
    Slot,
)
from onelife.belief_law.full_evaluation import FullStateScorer
from onelife.belief_law.predictive_cache import (
    MeanTableCachedScorer,
    MeanTablePredictor,
)
from onelife.belief_law.repair import BackoffCatchAll


def mass(outcomes):
    result = defaultdict(float)
    for o in outcomes:
        result[tuple(sorted(o.edits.items()))] += o.probability
    return dict(result)


def test_merge_and_cache_preserve_mass_and_invalidate_on_new_input():
    calls = []

    def predict(state, action):
        calls.append(action)
        return [
            Outcome({"x": state["x"]}, 0.2),
            Outcome({"x": state["x"] + 1}, 0.5),
            Outcome({"x": state["x"]}, 0.3),
        ]

    scorer = SimpleNamespace(active_input_key="first")
    wrapper = MeanTablePredictor(SimpleNamespace(predict=predict), scorer)
    expected = mass(predict({"x": 1}, "noop"))
    actual = wrapper.predict({"x": 1}, "noop")
    assert mass(actual) == pytest.approx(expected)
    assert len(actual) == 2
    assert wrapper.predict({"x": 1}, "noop") is actual
    assert len(calls) == 2
    scorer.active_input_key = "second"
    assert mass(wrapper.predict({"x": 5}, "noop")) == pytest.approx(
        {(("x", 5),): 0.5, (("x", 6),): 0.5}
    )


@pytest.mark.parametrize("table_type", [BackoffCatchAll, CategoricalCatchAll])
def test_component_cache_preserves_fixed_seed_predictions(table_type):
    table = table_type(
        frozenset({"player.hunger"}),
        lambda s, a: a,
        lambda s, edits: tuple(edits.items()),
        lambda s, sig: dict(sig),
    )
    table.observe_support({"player.hunger": 0}, "noop", {"player.hunger": 1})
    posterior = FactoredLawPosterior(
        [
            Slot(
                "survival",
                None,
                table.writes,
                frozenset(),
                [Candidate("catchall", table.predict, kind="catchall")],
            )
        ]
    )
    posterior.catchalls[0] = table

    class Model:
        def __init__(self, posterior):
            self.posterior = posterior

        def sample_next_state(self, state, action):
            nxt = self.posterior.sample_next_state(
                {"player.hunger": state.player.hunger}, action
            )
            state.player.hunger = nxt["player.hunger"]
            return state

    plain = FullStateScorer(Model(copy.deepcopy(posterior)), samples=2)
    cached = MeanTableCachedScorer(Model(copy.deepcopy(posterior)), samples=2)
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    for seed in range(10):
        assert plain.draw(state, "noop", seed) == cached.draw(state, "noop", seed)
    assert len(cached.component_events) == int(table_type is BackoffCatchAll)
