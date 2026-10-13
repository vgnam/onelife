"""Goal-conditioned closed loop reusing the acquisition law posterior."""

from onelife.belief_law.core import RootSamplingPlanner, State
from onelife.belief_law.revision import LibraryReviser


class BeliefLawAgent:
    def __init__(
        self, planner: RootSamplingPlanner, reviser: LibraryReviser | None = None
    ):
        self.planner, self.reviser = planner, reviser
        self.last_plan: dict | None = None

    def act(self, state: State) -> str:
        action, self.last_plan = self.planner.plan(state)
        return action

    def observe(self, state: State, action: str, nxt: State) -> None:
        self.planner.posterior.observe(state, action, nxt)
        if self.reviser is not None:
            self.reviser.run(self.planner.posterior)
