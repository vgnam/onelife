"""Cache frozen OneLife expert predictions while preserving its hidden RNG input.

The saved laws use only per-state RNG, never global RNG, when constructing
expert predictions. Sampling the combined distributions remains unchanged.
"""

import json
import pickle
from hashlib import sha256

from onelife.belief_law.predictive_cache import MeanTableCachedScorer


class OneLifeCachedScorer(MeanTableCachedScorer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.expert_cache = {}
        original = model._get_law_predictions

        def predictions(state, action):
            rng = state.__dict__.get("random_state")
            rng_key = (
                sha256(pickle.dumps(rng.get_state())).hexdigest()
                if rng is not None
                else None
            )
            # Preserve actual object/chunk order, duplicated player objects,
            # and both serialized and mutable cached RNG representations.
            visible = json.dumps(
                state.model_dump(
                    mode="json", exclude={"event_bus", "serialized_random_state"}
                ),
                sort_keys=True,
            )
            key = (
                sha256(visible.encode()).hexdigest(),
                action,
                state.serialized_random_state,
                rng_key,
            )
            if key not in self.expert_cache:
                self.expert_cache[key] = original(state, action)
            return self.expert_cache[key]

        model._get_law_predictions = predictions

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "expert_prediction_cache_entries": len(self.expert_cache),
        }
