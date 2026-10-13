from crafter_oo.functional_env import initial_state

from onelife.belief_law.core import Candidate, FactoredLawPosterior, Outcome, Slot
from onelife.belief_law.crafter import CrafterAdapter, CrafterBeliefWorldModel, changed
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.our_method.core import LawFunctionWrapper


def test_adapter_roundtrip_preserves_benchmark_observables():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    encoded = adapter.encode(state)
    assert _gamestate_to_json(adapter.decode(encoded, state)) == _gamestate_to_json(
        state
    )
    large = initial_state(area=(32, 32), view=(9, 9), seed=2)
    other = CrafterAdapter(large)
    assert _gamestate_to_json(other.decode(encoded)) == _gamestate_to_json(state)


def test_relative_catchall_transfers_numeric_deltas_and_target_tiles():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    before = adapter.encode(state)
    after = dict(before)
    x = cast(int, before["player.position.x"])
    y = cast(int, before["player.position.y"])
    after["player.inventory.wood"] = cast(int, after["player.inventory.wood"]) + 1
    after[f"tile.{x + 1}.{y}"] = "path"
    signature = adapter.signature(before, changed(before, after))
    moved = dict(before)
    moved["player.position.x"] = cast(int, moved["player.position.x"]) - 1
    moved["player.inventory.wood"] = 7
    edits = adapter.restore(moved, signature)
    assert edits["player.inventory.wood"] == 8
    assert edits[f"tile.{x}.{y}"] == "path"


def test_program_predictions_do_not_mutate_observed_state():
    class Law:
        def precondition(self, current_state, action):
            return action == "Do"

        def effect(self, current_state, action):
            current_state.player.inventory.wood += 1

    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    before = adapter.encode(state)
    wrapper = LawFunctionWrapper(Law(), "")
    outcomes = adapter.predictor(wrapper)(before, "do")
    assert outcomes[0].edits["player.inventory.wood"] == 1
    assert state.player.inventory.wood == 0
    assert before["player.inventory.wood"] == 0


def test_native_crafter_joint_law_is_evaluatable():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    slot = Slot(
        "wood",
        frozenset({"do"}),
        frozenset({"player.inventory.wood"}),
        frozenset({"player.inventory.wood"}),
        [
            Candidate(
                "wood",
                lambda s, a: [
                    Outcome(
                        {
                            "player.inventory.wood": cast(
                                int, s["player.inventory.wood"]
                            )
                            + 1
                        }
                    )
                ],
            )
        ],
    )
    model = CrafterBeliefWorldModel(FactoredLawPosterior([slot]), adapter)
    nxt = model.sample_next_state(state, "do")
    assert nxt.player.inventory.wood == 1
    assert model.evaluate_log_probability(state=state, action="do", next_state=nxt) == 0


def test_catchall_context_canonicalizes_acquisition_and_evaluator_actions():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    flat = adapter.encode(state)
    assert adapter.context(flat, "Do") == adapter.context(flat, "do")
    assert adapter.context(flat, "Move East") == adapter.context(flat, "move_right")


def test_incomplete_entity_birth_is_discarded_and_logged():
    state = initial_state(area=(9, 9), view=(9, 9), seed=1)
    adapter = CrafterAdapter(state)
    flat = adapter.encode(state)
    flat["entity.cow.999.health"] = 3
    recovered = adapter.decode(flat, state)
    assert not any(e.entity_id == 999 for e in recovered.objects)
    assert adapter.approximations["discarded_incomplete_entity_births"] == 1


from typing import cast
