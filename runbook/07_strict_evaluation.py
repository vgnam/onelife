"""Evaluate frozen models with the original full-state correctness criterion.

uv run python runbook/07_strict_evaluation.py
Runs native likelihoods and a shared-likelihood control. No fitting or LLM calls.
"""

import csv
import json
import math
import random
import runpy
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import cast

import cloudpickle
import numpy as np
import torch
from crafter_oo.functional_env import EnvConfig
from crafter_oo.state_export import WorldState
from loguru import logger

from onelife.belief_law.crafter import CrafterBeliefWorldModel
from onelife.belief_law.evaluation import MatchedObservableModel, isolated_randomness
from onelife.evaluator import EvaluationConfig, Evaluator
from onelife.evaluator.baselines import TrueTransitionWorldModel
from onelife.evaluator.core import EvaluatableWorldModel
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.serde import evaluation_results_to_jsonable
from onelife.our_method.crafter.observable_extractor import ObservableExtractor
from onelife.our_method.crafter.serde import load_law_mixture


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seed_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NativeScorer:
    """Preserve native likelihood while controlling the prediction RNG stream."""

    def __init__(self, draws: MatchedObservableModel):
        self.draws = draws
        self.scored_candidates = 0
        self.negative_infinity_scores = 0

    def sample_next_state(self, current_state: WorldState, action: str) -> WorldState:
        return self.draws.sample_next_state(current_state, action)

    def evaluate_log_probability(
        self, state: WorldState, action: str, next_state: WorldState
    ) -> float:
        # Candidate-independent seed: no held-out information enters randomness.
        seed = self.draws.draw_seed(self.draws.key(state, action), "native-ranking", 0)
        with isolated_randomness(self.draws.model, seed):
            model = cast(EvaluatableWorldModel[WorldState, str], self.draws.model)
            score = float(model.evaluate_log_probability(state, action, next_state))
        if math.isnan(score) or score == math.inf:
            raise ValueError("Native likelihood returned NaN or positive infinity")
        self.scored_candidates += 1
        self.negative_infinity_scores += score == -math.inf
        return score

    def diagnostics(self) -> dict:
        return {
            "scored_candidates": self.scored_candidates,
            "negative_infinity_scores": self.negative_infinity_scores,
            "nan_or_positive_infinity_scores": 0,
        }


class RecordedPredictions:
    def __init__(self, scorer):
        self.scorer = scorer
        self.hashes = []

    def sample_next_state(self, current_state: WorldState, action: str) -> WorldState:
        state = self.scorer.sample_next_state(current_state, action)
        self.hashes.append(
            sha256(
                json.dumps(_gamestate_to_json(state), sort_keys=True).encode()
            ).hexdigest()
        )
        return state

    def evaluate_log_probability(
        self, state: WorldState, action: str, next_state: WorldState
    ) -> float:
        return self.scorer.evaluate_log_probability(state, action, next_state)


def cache_matches(folder: Path, fingerprint: dict, oracle: bool) -> bool:
    manifest = folder / (
        "true_simulator_bank_manifest.json"
        if oracle
        else "predictive_bank_manifest.json"
    )
    if not manifest.exists():
        return False
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    if any(stored.get(key) != value for key, value in fingerprint.items()):
        return False
    if oracle:
        sources = stored.get("oracle_source_sha256", {})
        if not sources or any(
            not Path(name).exists()
            or sha256(Path(name).read_bytes()).hexdigest() != value
            for name, value in sources.items()
        ):
            return False
    return True


def main(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    helpers = runpy.run_path("runbook/06_matched_evaluation.py")
    logger.remove()
    logger.add(lambda message: print(message, end="", flush=True), level="INFO")
    logger.add(output / "run.log", level="INFO")
    start = perf_counter()
    posterior_path = args.belief_dir / "fitted_belief_law.pkl"
    with posterior_path.open("rb") as file:
        posterior, adapter = cloudpickle.load(file)
    one = load_law_mixture(args.onelife_model)
    extractor = one.observable_extractor
    if not isinstance(extractor, ObservableExtractor):
        raise TypeError(
            "Strict comparison requires the saved Crafter observable extractor"
        )
    if args.samples < 1:
        raise ValueError("Predictive sample budget must be positive")
    args.scorers = list(dict.fromkeys(args.scorers))
    historical = json.loads(args.onelife_results.read_text(encoding="utf-8"))
    config = historical["evaluation_config"]
    trials = args.trials or config["num_trials"]
    models = {
        "true_simulator": TrueTransitionWorldModel(
            helpers["true_simulator_step"],
            lambda left, right: _gamestate_to_json(left) == _gamestate_to_json(right),
        ),
        "belief_law": CrafterBeliefWorldModel(posterior, adapter),
        "belief_law_map": CrafterBeliefWorldModel(posterior, adapter, map_only=True),
        "onelife": one,
    }
    fingerprint = {
        "posterior_sha256": sha256(posterior_path.read_bytes()).hexdigest(),
        "onelife_sha256": sha256(args.onelife_model.read_bytes()).hexdigest(),
        "scorer_sha256": sha256(
            Path("src/onelife/belief_law/evaluation.py").read_bytes()
        ).hexdigest(),
        "sampler_source_sha256": {
            name: sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "src/onelife/belief_law/core.py",
                "src/onelife/belief_law/crafter.py",
                "src/onelife/our_method/world_modeling.py",
                "src/onelife/our_method/crafter/observable_extractor.py",
                "src/onelife/our_method/crafter/serde.py",
                "src/onelife/poe_world/core.py",
            ]
        },
        "seed": args.seed,
        "noise_logscore": args.noise_logscore,
    }
    seed_rng(args.seed)
    shared = CrafterEvaluationFactory(
        EnvConfig(size=adapter.template.size, view=adapter.template.view),
        policy_seed=config["eval_seed"],
    ).create_context(
        EvaluationConfig(num_distractors=config["num_distractors"], num_trials=trials),
        config["num_transitions_per_scenario"],
    )
    # None uses precisely the original evaluator's WorldState equality, rather
    # than the observable-equivalence override in the previous comparison.
    shared.state_equivalence = None
    if args.scenarios:
        shared.test_transitions = dict(
            list(shared.test_transitions.items())[: args.scenarios]
        )
    transitions = [t for ts in shared.test_transitions.values() for t in ts]
    results, diagnostics, timings, prediction_hashes = {}, {}, {}, {}
    reference_distractors = None
    transition_hashes = None
    for name, model in models.items():
        maximum = MatchedObservableModel(
            model,
            extractor,
            samples=args.samples,
            seed=args.seed,
            noise_logscore=args.noise_logscore,
        )
        hashes = [maximum.key(t.prev_metadata, t.action) for t in transitions]
        if transition_hashes is None:
            transition_hashes = hashes
        elif hashes != transition_hashes:
            raise RuntimeError("Models received different input transitions")
        if "shared" in args.scorers:
            cached = args.bank_dir / f"{name}_predictive_banks.json"
            if cached.exists() and cache_matches(
                args.bank_dir, fingerprint, name == "true_simulator"
            ):
                maximum.load_banks(cached)
                logger.info("{}: loaded verified predictive banks", name)
            tick = perf_counter()
            for t in transitions:
                maximum.predictive_statistics(t.prev_metadata, t.action)
            timings[f"{name}_predictive_bank_seconds"] = perf_counter() - tick
            maximum.save_banks(output / f"{name}_predictive_banks.json")
        for mode in args.scorers:
            draws = MatchedObservableModel(
                model,
                extractor,
                samples=args.samples,
                seed=args.seed,
                noise_logscore=args.noise_logscore,
                banks=maximum.banks,
            )
            scorer = NativeScorer(draws) if mode == "native" else draws
            recorded = RecordedPredictions(scorer)
            seed_rng(args.seed)
            context = helpers["paired_context"](shared, config["eval_seed"])
            assert context.state_equivalence is None
            distractors = helpers["RecordedDistractors"](context.distractor_generator)
            context.distractor_generator = distractors
            tick = perf_counter()
            logger.info(
                "Evaluating {} / {} with original full-state correctness", name, mode
            )
            result = evaluation_results_to_jsonable(
                Evaluator(context).evaluate(recorded)
            )
            if reference_distractors is None:
                reference_distractors = distractors.hashes
            elif reference_distractors != distractors.hashes:
                raise RuntimeError("Models or scorers received different distractors")
            if name in prediction_hashes and prediction_hashes[name] != recorded.hashes:
                raise RuntimeError("Scorer modes generated different predicted states")
            prediction_hashes[name] = recorded.hashes
            key = f"{name}_{mode}"
            results[key] = result
            diagnostics[key] = scorer.diagnostics()
            timings[f"{key}_seconds"] = perf_counter() - tick
            dump(output / f"{key}_results.json", result)
            logger.info(
                "{} accuracy={:.4f} MRR={:.4f} IoU={:.4f}",
                key,
                result["discriminative_accuracy"],
                result["reciprocal_rank"],
                result["edit_distance"]["intersection_over_union"],
            )
    assert reference_distractors is not None
    metadata = {
        **fingerprint,
        "candidate_equivalence": "original native WorldState equality; no observable equivalence",
        "scorers": args.scorers,
        "native_scoring": "original evaluate_log_probability of each frozen model; no added scoring noise",
        "shared_scoring": "same saved observable extractor, domains and smoothed MC marginal likelihood for every model",
        "shared_samples": args.samples,
        "num_scenarios": len(shared.test_transitions),
        "num_transitions": len(transitions),
        "num_trials": trials,
        "paired_transition_hashes": transition_hashes,
        "paired_distractor_batches": len(reference_distractors),
        "paired_distractor_sha256": sha256(
            json.dumps(reference_distractors).encode()
        ).hexdigest(),
        "predictions_identical_between_scorers": True,
        "prediction_sha256": {
            name: sha256(json.dumps(values).encode()).hexdigest()
            for name, values in prediction_hashes.items()
        },
        "evaluator_sha256": sha256(
            Path("src/onelife/evaluator/core.py").read_bytes()
        ).hexdigest(),
        "runbook_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": sha256(
            Path("runbook/06_matched_evaluation.py").read_bytes()
        ).hexdigest(),
        "llm_network_calls": 0,
        "fitting_updates": 0,
        "heldout_usage": "evaluation only",
        "timings": timings,
        "total_seconds": perf_counter() - start,
        "limits": [
            "Native likelihoods have different support: BELIEF-LAW scores full encoded states, OneLife scores its observable projection with original noise and persistence penalties.",
            "The shared-scorer control keeps the same probability scoring as step 6 and restores only the original full-state correctness criterion.",
            "Sampling streams are isolated and identical between modes; these are fresh paired runs, not copied historical results.",
            "True replays the complete previous state and serialized RNG; its copied mutable RNG cache is discarded.",
            "The fixed-library posterior is not updated, and LLM revision is not run.",
        ],
    }
    dump(
        output / "comparison.json",
        {"metadata": metadata, "results": results, "diagnostics": diagnostics},
    )
    rows = []
    for key, r in results.items():
        rows.append(
            {
                "model_scorer": key,
                "accuracy": r["discriminative_accuracy"],
                "mrr": r["reciprocal_rank"],
                "edit_distance": r["edit_distance"]["raw"],
                "patch_iou": r["edit_distance"]["intersection_over_union"],
            }
        )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_rows = []
    for scenario in shared.test_transitions:
        row = {"scenario": scenario}
        for key, r in results.items():
            m = r["metrics_by_source"][scenario]["mean"]
            row[f"{key}_accuracy"] = m["discriminative_accuracy"]
            row[f"{key}_mrr"] = m["reciprocal_rank"]
        scenario_rows.append(row)
    with (output / "by_scenario.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(scenario_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_rows)
    report = [
        "# Original full-state evaluation: True, BELIEF-LAW, MAP and OneLife",
        "",
        f"Frozen models; {len(shared.test_transitions)} scenarios, {len(transitions)} transitions, {trials} trials, seed {args.seed}. No fitting or LLM calls.",
        "Every run uses the original native evaluator correctness criterion. Candidates differing only in unscored fields are no longer accepted as observable equivalents.",
        "",
    ]
    for mode in args.scorers:
        report += [
            "## "
            + (
                "Original native likelihoods"
                if mode == "native"
                else "Shared likelihood control, original full-state correctness"
            ),
            "",
            "| Model | Accuracy ↑ | MRR ↑ | Edit distance ↓ | Patch IoU ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in models:
            r = results[f"{name}_{mode}"]
            report.append(
                f"| {name} | {r['discriminative_accuracy']:.2%} | {r['reciprocal_rank']:.4f} | {r['edit_distance']['raw']:.4f} | {r['edit_distance']['intersection_over_union']:.4f} |"
            )
        report.append("")
    report += [
        f"Identical distractors verified over {len(reference_distractors)} batches for every model/scorer. The same model generates identical predictions in both scorer modes.",
        "",
        "## Scope",
        "",
    ]
    report += ["- " + item for item in metadata["limits"]]
    (output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("Saved strict comparison to {}", output / "comparison.md")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--belief-dir", type=Path, default=Path("runbook_output/04_belief_law")
    )
    parser.add_argument(
        "--onelife-model",
        type=Path,
        default=Path("runbook_output/03_glm/fitted_world_model.json"),
    )
    parser.add_argument(
        "--onelife-results",
        type=Path,
        default=Path("runbook_output/03_glm/evaluation_results.json"),
    )
    parser.add_argument(
        "--bank-dir", type=Path, default=Path("runbook_output/06_matched_evaluation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runbook_output/07_strict_evaluation")
    )
    parser.add_argument(
        "--scorers",
        choices=["native", "shared"],
        nargs="+",
        default=["native", "shared"],
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--noise-logscore", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=0)
    main(parser.parse_args())
