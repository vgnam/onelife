"""Audit held-out predictive support of the fitted BELIEF-LAW library.

This reads held-out scenarios only after fitting; it never feeds this data into
acquisition, proposals, or posterior updates.
"""

import json
import random
from collections import Counter
from pathlib import Path

import cloudpickle
import numpy as np
from crafter_oo.functional_env import EnvConfig
from loguru import logger

from onelife.belief_law.core import matches
from onelife.evaluator import EvaluationConfig
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory


def main():
    logger.remove()
    folder = Path("runbook_output/04_belief_law")
    with (folder / "fitted_belief_law.pkl").open("rb") as file:
        posterior, adapter = cloudpickle.load(file)
    random.seed(42)
    np.random.seed(42)
    factory = CrafterEvaluationFactory(
        EnvConfig(size=adapter.template.size, view=adapter.template.view),
        policy_seed=42,
    )
    context = factory.create_context(
        EvaluationConfig(num_trials=1, num_distractors=10), 30
    )
    blocked = Counter()
    finite, total = 0, 0
    sources = {}
    for name, transitions in context.test_transitions.items():
        failed = 0
        for transition in transitions:
            state, nxt = (
                adapter.encode(transition.prev_metadata),
                adapter.encode(transition.next_metadata),
            )
            adapter.remember(state, transition.prev_metadata)
            action = transition.action
            total += 1
            probability = posterior.evaluate_log_probability(state, action, nxt)
            if np.isfinite(probability):
                finite += 1
            else:
                failed += 1
            for component in posterior.components:
                writes = frozenset(
                    s
                    for i in component.slots
                    if posterior.slots[i].compatible(action)
                    for s in posterior.slots[i].writes
                )
                variables = {v for v in state.keys() | nxt.keys() if matches(v, writes)}
                if not variables:
                    continue
                cache = {("variables", component.slots): variables}
                mass = sum(
                    pz
                    * sum(
                        p
                        for _, p, _ in posterior._events(
                            component, z, state, action, cache, observed=nxt
                        )
                    )
                    for z, pz in zip(component.assignments, component.probabilities)
                    if pz > 0
                )
                if mass == 0:
                    blocked[", ".join(sorted(writes))] += 1
        sources[name] = {"transitions": len(transitions), "unsupported_truth": failed}
    report = {
        "total_transitions": total,
        "truth_with_finite_predictive_probability": finite,
        "unsupported_truth": total - finite,
        "blocking_component_write_sets": dict(blocked.most_common()),
        "by_scenario": sources,
        "interpretation": "Full-state posterior predictive has a closed relative-edit dictionary. Unknown contexts/deltas can assign zero mass to true held-out states. OneLife instead scores a subset of observables with expanded-support noise and soft persistence penalties; ranking results therefore mix model quality with scoring support.",
        "heldout_usage": "Audit only; no fitting or law revision from these transitions",
    }
    (folder / "predictive_support_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "by_scenario"}, indent=2))


if __name__ == "__main__":
    main()
