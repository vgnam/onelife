import pickle
import runpy
from pathlib import Path

from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.evaluator.crafter.mutators.entity_health import PlayerHealthMutator
from onelife.evaluator.crafter.scenarios import CowMovementScenario, run_scenarios


def test_true_replay_uses_saved_rng_after_distractor_mutation():
    script = Path(__file__).resolve().parents[2] / "runbook/06_matched_evaluation.py"
    replay = runpy.run_path(str(script))["true_simulator_step"]
    transitions = run_scenarios([CowMovementScenario(max_steps=10)])[0].transitions
    for transition in transitions:
        state = transition.prev_metadata
        before = _gamestate_to_json(state)
        for _ in range(10):
            PlayerHealthMutator()(state, transition.action)
        cached_rng = pickle.dumps(state.random_state.get_state())
        predicted = replay(state, transition.action)
        assert _gamestate_to_json(predicted) == _gamestate_to_json(
            transition.next_metadata
        )
        assert _gamestate_to_json(state) == before
        assert pickle.dumps(state.random_state.get_state()) == cached_rng
