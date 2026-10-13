import copy
import math
import pickle
import random
import runpy
from pathlib import Path

import numpy as np
import torch
from crafter_oo.functional_env import initial_state

from onelife.belief_law.evaluation import MatchedObservableModel
from onelife.our_method.crafter.observable_extractor import ObservableExtractor


class NativeTestModel:
    def __init__(self):
        self.rng_draws = []

    def sample_next_state(self, current_state, action):
        current_state.player.inventory.wood = random.randint(0, 5)
        return current_state

    def evaluate_log_probability(self, state, action, next_state):
        self.rng_draws.append(
            (random.random(), float(np.random.uniform()), float(torch.rand(())))
        )
        return 0.0 if next_state.player.inventory.wood == 1 else -math.inf


def native_scorer(draws):
    script = Path(__file__).resolve().parents[2] / "runbook/07_strict_evaluation.py"
    return runpy.run_path(str(script))["NativeScorer"](draws)


def test_native_scores_keep_zero_and_negative_infinity_and_restore_rng():
    model = NativeTestModel()
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    candidate = copy.deepcopy(state)
    candidate.player.inventory.wood = 1
    scorer = native_scorer(MatchedObservableModel(model, ObservableExtractor()))
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())
    torch_rng = torch.random.get_rng_state()
    assert scorer.evaluate_log_probability(state, "do", candidate) == 0
    candidate.player.inventory.wood = 2
    assert scorer.evaluate_log_probability(state, "do", candidate) == -math.inf
    assert model.rng_draws[0] == model.rng_draws[1]
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert scorer.diagnostics()["negative_infinity_scores"] == 1


def test_native_and_shared_modes_use_identical_generation_streams():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    shared = MatchedObservableModel(NativeTestModel(), ObservableExtractor())
    native = native_scorer(
        MatchedObservableModel(NativeTestModel(), ObservableExtractor())
    )
    for _ in range(3):
        assert native.sample_next_state(state, "do") == shared.sample_next_state(
            state, "do"
        )
