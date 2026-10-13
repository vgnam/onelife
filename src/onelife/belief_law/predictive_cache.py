"""Evaluation-only memoization of deterministic, integrated catch-all means.

Never cache program effects, Dirichlet draws, or stochastic escape proposals.
The fitted posterior is frozen during evaluation. Identical emitted edit maps
are merged by summing mass; the predictive distribution is preserved.
"""

from onelife.belief_law.core import Outcome
from onelife.belief_law.full_evaluation import FullStateScorer


class MeanTablePredictor:
    def __init__(self, table, scorer, merge=True):
        self.table, self.scorer, self.merge = table, scorer, merge
        self.cache = {}

    def predict(self, state, action):
        key = self.scorer.active_input_key
        if key not in self.cache:
            if not self.merge:
                self.cache[key] = self.table.predict(state, action)
                return self.cache[key]
            merged = {}
            for outcome in self.table.predict(state, action):
                signature = tuple(sorted(outcome.edits.items()))
                merged[signature] = merged.get(signature, 0.0) + outcome.probability
            self.cache[key] = [Outcome(dict(s), p) for s, p in merged.items()]
        return self.cache[key]


class MeanTableCachedScorer(FullStateScorer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.active_input_key = None
        self.mean_tables = []
        self.component_events = {}
        posterior = getattr(model, "posterior", None)
        if posterior is not None:
            for i, table in posterior.catchalls.items():
                if hasattr(table, "base_sample"):
                    continue
                wrapper = MeanTablePredictor(
                    table, self, merge=getattr(table, "integrated_probabilities", False)
                )
                for candidate in posterior.slots[i].candidates:
                    if candidate.kind == "catchall":
                        candidate.predict = wrapper.predict
                self.mean_tables.append(wrapper)
            for slot in posterior.slots:
                for candidate in slot.candidates:
                    source = candidate.source.lower()
                    if (
                        candidate.kind == "program"
                        and source
                        and not any(
                            token in source
                            for token in (
                                "random",
                                "rand(",
                                "randint",
                                "choice",
                                "sample",
                                "shuffle",
                                "self.",
                            )
                        )
                    ):
                        original_predict = candidate.predict
                        # Programs with no RNG or self-state access are pure
                        # input-to-outcome functions. Preserve their outcome
                        # order and probabilities; Dirichlet draws stay fresh.
                        wrapper = MeanTablePredictor(
                            type(
                                "Predictor",
                                (),
                                {"predict": staticmethod(original_predict)},
                            )(),
                            self,
                            merge=False,
                        )
                        candidate.predict = wrapper.predict
                        self.mean_tables.append(wrapper)
            original_events = getattr(
                posterior, "_evaluation_uncached_events", posterior._events
            )
            posterior._evaluation_uncached_events = original_events

            def events(
                component,
                assignment,
                state,
                action,
                cache=None,
                parameter_sample=None,
                observed=None,
            ):
                assert posterior is not None
                deterministic = observed is None and all(
                    not posterior.slots[i].candidates[k].learn_probabilities
                    and (
                        posterior.slots[i].candidates[k].kind == "null"
                        or posterior.slots[i].candidates[k].name.startswith("Train")
                        or posterior.slots[i].candidates[k].kind == "catchall"
                        and getattr(
                            posterior.catchalls[i], "integrated_probabilities", False
                        )
                        and not hasattr(posterior.catchalls[i], "base_sample")
                    )
                    for i, k in zip(component.slots, assignment)
                    if posterior.slots[i].compatible(action)
                )
                key = self.active_input_key, component.slots, assignment
                if deterministic and key in self.component_events:
                    return self.component_events[key]
                result = original_events(
                    component,
                    assignment,
                    state,
                    action,
                    cache,
                    parameter_sample,
                    observed,
                )
                if deterministic:
                    self.component_events[key] = result
                return result

            posterior._events = events

    def draw(self, state, action, seed):
        self.active_input_key = self.key(state, action)
        return super().draw(state, action, seed)

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "evaluation_optimization": "merged deterministic catch-all means",
            "mean_table_cache_entries": sum(len(t.cache) for t in self.mean_tables),
            "component_event_cache_entries": len(self.component_events),
        }
