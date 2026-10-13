from ....typing_utils import implements
from .interface import Mutator
from crafter_oo.state_export import WorldState
from crafter_oo.constants import ActionT
import random


class PlayerHealthMutator:
    def __init__(self):
        self.category = "Combat"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        # Do not draw from `state.random_state`: it is a cached property, so
        # drawing from it advances the input state's RNG, which world models
        # read when predicting. Range matches the former numpy randint(-3, 3).
        delta = random.randint(-3, 2)
        mutated_state.player.health = mutated_state.player.health + delta

        # Clamp health between 0 and 10
        mutated_state.player.health = max(0, min(mutated_state.player.health, 10))

        return mutated_state


implements(Mutator)(PlayerHealthMutator)


class EntityHealthMutator:
    def __init__(self):
        self.category = "Combat"

    def precondition(self, state: WorldState, action: ActionT) -> bool:
        return True

    def __call__(self, state: WorldState, action: ActionT) -> WorldState:
        mutated_state = state.model_copy(deep=True)

        for entity in mutated_state.objects:
            if entity.entity_id == mutated_state.player.entity_id:
                continue

            # Choose a random value for the health that is not the
            # current health or a value that is within 1 of the current health
            values = {_ for _ in range(0, 11)}
            values = values - {entity.health, entity.health - 1, entity.health + 1}

            entity.health = random.choice(list(values))

        return mutated_state


implements(Mutator)(EntityHealthMutator)
