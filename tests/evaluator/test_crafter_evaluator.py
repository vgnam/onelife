import pickle
import rich
from crafter_oo.functional_env import EnvConfig
from crafter_oo.functional_env import transition as crafter_transition_fn
from crafter_oo.state_export import WorldState

from onelife.evaluator.baselines import NullWorldModel, TrueTransitionWorldModel
from onelife.evaluator.core import EvaluationConfig, Evaluator
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
import pytest


@pytest.mark.flaky(retries=5, delay=0)
def test_evaluating_true_vs_null_world_model():
    """
    Test that the evaluation framework can successfully evaluate a world model
    and produce meaningful results using the factory and actual evaluator.
    """
    env_config = EnvConfig(size=(9, 9), view=(9, 9))
    factory = CrafterEvaluationFactory(env_config, policy_seed=42)
    config = EvaluationConfig(num_distractors=2)

    context = factory.create_context(config, num_transitions_per_scenario=30)

    def equality_check(state1: WorldState, state2: WorldState) -> bool:
        return _gamestate_to_json(state1) == _gamestate_to_json(state2)

    def wrap_true_transition_fn(state: WorldState, action) -> WorldState:
        next_state, _ = crafter_transition_fn(state, MAP_ACTION_TO_INDEX[action])
        return next_state

    true_model = TrueTransitionWorldModel(
        wrap_true_transition_fn, equal_fn=equality_check
    )
    null_model = NullWorldModel(equality_check)

    evaluator = Evaluator(context)

    true_wm_perf = evaluator.evaluate(true_model)

    null_wm_perf = evaluator.evaluate(null_model)

    assert (
        true_wm_perf.discriminative_accuracy == 1.0
    ), "True transition model should have accuracy greater than 0.5"

    #
    assert (
        true_wm_perf.edit_distance.raw == 0.0
    ), "True transition model should have low generative error"

    # Null model should perform worse than true model
    assert (
        null_wm_perf.discriminative_accuracy < true_wm_perf.discriminative_accuracy
    ), "Null model should perform worse than true model"

    assert (
        null_wm_perf.normalized_recall < true_wm_perf.normalized_recall
    ), "Null model should perform worse than true model"

    rich.print("== True World Model Performance ==")
    rich.print(true_wm_perf)

    rich.print("== Null World Model Performance ==")
    rich.print(null_wm_perf)


def test_context_keeps_every_scenario():
    """Each scenario gets its own key; successful and unsuccessful variants of
    a mechanic must not overwrite each other."""
    factory = CrafterEvaluationFactory(EnvConfig(size=(9, 9), view=(9, 9)))
    context = factory.create_context(
        EvaluationConfig(num_distractors=2), num_transitions_per_scenario=30
    )

    assert len(context.test_transitions) == 38
    coal = context.test_transitions["collect_coal"][0]
    no_coal = context.test_transitions["collect_coal_unsuccessful"][0]
    assert coal.next_metadata.player.inventory.coal == 1
    assert no_coal.next_metadata.player.inventory.coal == 0


def test_context_caps_transitions_per_scenario():
    factory = CrafterEvaluationFactory(EnvConfig(size=(9, 9), view=(9, 9)))
    context = factory.create_context(
        EvaluationConfig(num_distractors=2), num_transitions_per_scenario=2
    )

    assert max(map(len, context.test_transitions.values())) == 2


def test_distractor_generation_does_not_advance_input_rng():
    """World models read `prev_metadata.random_state`; generating distractors
    must not change it, or later trials see a different input."""
    factory = CrafterEvaluationFactory(EnvConfig(size=(9, 9), view=(9, 9)))
    context = factory.create_context(
        EvaluationConfig(num_distractors=10), num_transitions_per_scenario=30
    )
    transition = context.test_transitions["cow_movement"][0]
    all_transitions = [
        t for transitions in context.test_transitions.values() for t in transitions
    ]
    before = pickle.dumps(transition.prev_metadata.random_state.get_state())

    context.distractor_generator(transition, all_transitions, num_distractors=10)

    after = pickle.dumps(transition.prev_metadata.random_state.get_state())
    assert before == after
