import copy
import pickle
import random
from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest
import torch
from crafter_oo.functional_env import initial_state
from crafter_oo.state_export import WorldState

from onelife.belief_law.evaluation import ABSENT, OTHER, MatchedObservableModel
from onelife.evaluator.core import (
    EvaluationConfig,
    EvaluationContext,
    Evaluator,
    SymbolicTransition,
)
from onelife.evaluator.crafter.components import JSONPatchEditDistance
from onelife.our_method.crafter.observable_extractor import ObservableExtractor


class TestSampler:
    __test__ = False

    def __init__(self, effect: Callable[[WorldState], None] | None = None):
        self.effect = effect
        self.calls = 0

    def sample_next_state(self, current_state: WorldState, action: str) -> WorldState:
        self.calls += 1
        if self.effect is not None:
            self.effect(current_state)
        return current_state


def test_shared_projection_ignores_unscored_fields_for_both_models():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    candidate = copy.deepcopy(state)
    candidate.daylight = 0.12345
    candidate.step_count += 9
    candidate.player.hunger += 5
    first = MatchedObservableModel(TestSampler(), ObservableExtractor(), samples=4)
    second = MatchedObservableModel(TestSampler(), ObservableExtractor(), samples=4)
    score = first.evaluate_log_probability(state, "do", candidate)
    assert score == first.evaluate_log_probability(state, "do", state)
    assert score == second.evaluate_log_probability(state, "do", candidate)
    assert first.nonfinite_scores == 0


def test_noise_normalizes_marginals_and_supports_unseen_values():
    scorer = MatchedObservableModel(TestSampler(), ObservableExtractor(), samples=4)
    name = "entity_99_health"
    counts = Counter({3: 2})  # Absent in half of the predictive samples.
    probabilities = {
        v: scorer.marginal_probability(name, counts, v) for v in scorer.domain(name)
    }
    assert sum(probabilities.values()) == pytest.approx(1)
    assert probabilities[ABSENT] > probabilities[4] > 0
    assert scorer.marginal_probability(name, counts, 999) == probabilities[OTHER]
    assert np.isfinite(np.log(probabilities[4]))
    exists = "entity_exists_99"
    assert scorer.marginal_probability(
        exists, Counter(), 0
    ) > scorer.marginal_probability(exists, Counter(), 1)
    assert sum(
        scorer.marginal_probability(exists, Counter(), v) for v in scorer.domain(exists)
    ) == pytest.approx(1)
    assert sum(
        scorer.marginal_probability("player_health", Counter(), v)
        for v in scorer.domain("player_health")
    ) == pytest.approx(1)


def test_shared_score_prefers_predicted_observable_without_candidate_leakage():
    def gain_wood(state):
        state.player.inventory.wood += 1

    sampler = TestSampler(gain_wood)
    scorer = MatchedObservableModel(sampler, ObservableExtractor(), samples=8)
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    original = copy.deepcopy(state)
    correct = copy.deepcopy(state)
    correct.player.inventory.wood += 1
    wrong = copy.deepcopy(state)
    wrong.player.inventory.wood = 999
    assert scorer.evaluate_log_probability(
        state, "do", correct
    ) > scorer.evaluate_log_probability(state, "do", wrong)
    assert sampler.calls == 8  # All candidate evaluations reuse the same bank.
    assert state == original
    assert scorer.outside_domain_hits == 1


def test_sampling_restores_evaluator_rng_and_budgets_share_nested_bank():
    def move(state):
        state.player.position.x = int(np.random.randint(0, 9))
        random.random()
        torch.rand(1)

    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    sampler = TestSampler(move)
    first = MatchedObservableModel(sampler, ObservableExtractor(), samples=8)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        pickle.dumps(np.random.get_state()),
        torch.get_rng_state(),
    )
    generated = first.sample_next_state(state, "do")
    first.predictive_statistics(state, "do")
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
    second = MatchedObservableModel(
        sampler, ObservableExtractor(), samples=4, banks=first.banks
    )
    assert second.sample_next_state(state, "do") == generated
    calls = sampler.calls
    second.predictive_statistics(state, "do")
    assert sampler.calls == calls
    bank = first.banks[first.key(state, "do")]
    expected = Counter(o["player_position_x"] for o in bank.observations[:4])
    assert second.predictive_statistics(state, "do")["player_position_x"] == expected


def test_domains_and_smoothing_do_not_depend_on_candidates():
    scorer = MatchedObservableModel(TestSampler(), ObservableExtractor(), samples=4)
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    candidate = copy.deepcopy(state)
    candidate.player.health = 500
    domain = scorer.domain("player_health")
    assert np.isfinite(scorer.evaluate_log_probability(state, "do", candidate))
    assert scorer.domain("player_health") == domain
    assert 500 not in domain
    assert OTHER in domain


def test_predictive_bank_key_canonicalizes_object_and_chunk_order():
    scorer = MatchedObservableModel(TestSampler(), ObservableExtractor(), samples=4)
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    reordered = copy.deepcopy(state)
    reordered.objects.reverse()
    reordered.chunks.reverse()
    for chunk in reordered.chunks:
        chunk.objects.reverse()
    assert scorer.key(state, "do") == scorer.key(reordered, "do")


def test_evaluator_observable_equivalence_is_optional_and_respects_observed_errors():
    def tick(state):
        state.step_count += 1

    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    truth = copy.deepcopy(state)
    truth.daylight = 0.12345
    transition = SymbolicTransition(state, "do", truth)
    extractor = ObservableExtractor()
    scorer = MatchedObservableModel(TestSampler(tick), extractor, samples=4)
    context = EvaluationContext(
        test_transitions={"example": [transition]},
        distractor_generator=lambda transition, all_transitions, num_distractors: [],
        edit_distance_calculator=JSONPatchEditDistance(),
        config=EvaluationConfig(num_distractors=0),
    )
    random.seed(1)  # Predicted candidate is first after shuffling the tied pair.
    native = Evaluator(context)._evaluate_transition(scorer, transition, [transition])
    assert native.discriminative_accuracy == 0
    context.state_equivalence = lambda left, right: (
        extractor.get_observed_outcomes(left) == extractor.get_observed_outcomes(right)
    )
    random.seed(1)
    matched = Evaluator(context)._evaluate_transition(scorer, transition, [transition])
    assert matched.discriminative_accuracy == 1
    assert matched.reciprocal_rank == 1
    assert matched.normalized_recall == 1
    truth.player.inventory.wood += 1
    random.seed(1)
    wrong = Evaluator(context)._evaluate_transition(scorer, transition, [transition])
    assert wrong.discriminative_accuracy == 0
