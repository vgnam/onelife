"""
Factory for creating evaluation contexts for the Crafter environment.
"""

from crafter_oo.state_export import WorldState
from crafter_oo.functional_env import transition, initial_state, EnvConfig
import random

from ..core import EvaluationContext, EvaluationConfig, SymbolicTransition
from .components import JSONPatchEditDistance, CrafterDistractorGenerator
from .scenarios import (
    run_scenarios,
    CraftWoodenPickaxeScenario,
    CowMovementScenario,
    RandomMovementScenario,
    ZombieDefeatScenario,
    DefeatSkeletonScenario,
    EatCowScenario,
    CollectCoalScenario,
    UnsuccessfulCollectCoalScenario,
    CollectDiamondScenario,
    UnsuccessfulCollectDiamondScenario,
    CollectIronScenario,
    UnsuccessfulCollectIronScenario,
    CollectStoneScenario,
    UnsuccessfulCollectStoneScenario,
    CollectDrinkScenario,
    CollectWoodScenario,
    EatPlantScenario,
    UnsuccessfulEatPlantScenario,
    CraftIronPickaxeScenario,
    UnsuccessfulCraftIronPickaxeScenario,
    CraftIronSwordScenario,
    UnsuccessfulCraftIronSwordScenario,
    CraftStonePickaxeScenario,
    UnsuccessfulCraftStonePickaxeScenario,
    CraftStoneSwordScenario,
    UnsuccessfulCraftStoneSwordScenario,
    UnsuccessfulCraftWoodenPickaxeScenario,
    CraftWoodenSwordScenario,
    UnsuccessfulCraftWoodenSwordScenario,
    PlaceFurnaceScenario,
    UnsuccessfulPlaceFurnaceScenario,
    PlacePlantScenario,
    UnsuccessfulPlacePlantScenario,
    PlaceStoneScenario,
    UnsuccessfulPlaceStoneScenario,
    PlaceTableScenario,
    UnsuccessfulPlaceTableScenario,
    WakeUpScenario,
)
from loguru import logger
from crafter_oo.constants import ActionT as CrafterAction


class CrafterEvaluationFactory:
    def __init__(self, env_config: EnvConfig, policy_seed: int = 42):
        self.env_config = env_config
        self.policy_seed = policy_seed
        self.policy_rng = random.Random(policy_seed)
        self.initial_state = initial_state(
            area=env_config.size,
            view=env_config.view,
            episode=1,
            seed=policy_seed,
        )
        self.transition_fn = transition

    def create_context(
        self, config: EvaluationConfig, num_transitions_per_scenario: int
    ) -> EvaluationContext[WorldState, CrafterAction]:

        # 1. Define and run scenarios to collect trajectories
        scenarios = [
            CraftWoodenPickaxeScenario(),
            CowMovementScenario(),
            RandomMovementScenario(max_steps=10),
            ZombieDefeatScenario(),
            DefeatSkeletonScenario(),
            EatCowScenario(),
            CollectCoalScenario(),
            UnsuccessfulCollectCoalScenario(),
            CollectDiamondScenario(),
            UnsuccessfulCollectDiamondScenario(),
            CollectIronScenario(),
            UnsuccessfulCollectIronScenario(),
            CollectStoneScenario(),
            UnsuccessfulCollectStoneScenario(),
            CollectDrinkScenario(),
            CollectWoodScenario(),
            EatPlantScenario(),
            UnsuccessfulEatPlantScenario(),
            CraftIronPickaxeScenario(),
            UnsuccessfulCraftIronPickaxeScenario(),
            CraftIronSwordScenario(),
            UnsuccessfulCraftIronSwordScenario(),
            CraftStonePickaxeScenario(),
            UnsuccessfulCraftStonePickaxeScenario(),
            CraftStoneSwordScenario(),
            UnsuccessfulCraftStoneSwordScenario(),
            UnsuccessfulCraftWoodenPickaxeScenario(),
            CraftWoodenSwordScenario(),
            UnsuccessfulCraftWoodenSwordScenario(),
            PlaceFurnaceScenario(),
            UnsuccessfulPlaceFurnaceScenario(),
            PlacePlantScenario(),
            UnsuccessfulPlacePlantScenario(),
            PlaceStoneScenario(),
            UnsuccessfulPlaceStoneScenario(),
            PlaceTableScenario(),
            UnsuccessfulPlaceTableScenario(),
            WakeUpScenario(),
        ]
        scenario_results = run_scenarios(scenarios)

        test_transitions: dict[
            str, list[SymbolicTransition[WorldState, CrafterAction]]
        ] = dict()
        for result in scenario_results:
            name = result.scenario.name
            # Scenarios are keyed by name, so a duplicate would silently
            # overwrite an earlier mechanic.
            if name in test_transitions:
                raise ValueError(f"Duplicate scenario name: {name}")
            if not result.goal_test:
                raise RuntimeError(
                    f"Scenario {name} did not reach its goal: {result.goal_test.message}"
                )
            test_transitions[name] = result.transitions[:num_transitions_per_scenario]

        logger.info(
            f"Collected {sum(map(len, test_transitions.values()))} test transitions"
            f" from {len(test_transitions)} scenarios"
        )

        # 2. Instantiate the distractor generator with all mutators
        distractor_generator = CrafterDistractorGenerator(seed=self.policy_seed)

        # 3. Assemble the context
        return EvaluationContext[WorldState, CrafterAction](
            config=config,
            test_transitions=test_transitions,
            distractor_generator=distractor_generator,  # type error here
            edit_distance_calculator=JSONPatchEditDistance(),
        )
