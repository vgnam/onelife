"""Compare frozen BELIEF-LAW and OneLife with one shared observable scorer.

uv run python runbook/06_matched_evaluation.py
No acquisition, fitting, law revision or network calls occur in this runbook.
"""

import copy
import csv
import json
import random
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import cast

import cloudpickle
import numpy as np
import torch
from crafter_oo.constants import ActionT as CrafterAction
from crafter_oo.functional_env import EnvConfig
from crafter_oo.functional_env import transition as crafter_transition
from crafter_oo.state_export import WorldState
from loguru import logger

from onelife.belief_law.crafter import CrafterBeliefWorldModel
from onelife.belief_law.evaluation import MatchedObservableModel
from onelife.evaluator import EvaluationConfig, Evaluator
from onelife.evaluator.baselines import TrueTransitionWorldModel
from onelife.evaluator.crafter.components import (
    CrafterDistractorGenerator,
    _gamestate_to_json,
)
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.crafter.utils import MAP_ACTION_TO_INDEX
from onelife.evaluator.serde import evaluation_results_to_jsonable
from onelife.our_method.crafter.observable_extractor import ObservableExtractor
from onelife.our_method.crafter.serde import load_law_mixture


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def true_simulator_step(state: WorldState, action: str) -> WorldState:
    """Replay the actual simulator from the input state, without target access.

    The simulator restores the input's environment RNG. This is a privileged
    oracle, not a learned model. Copy because transition increments step_count.
    """
    replay_state = copy.deepcopy(state)
    # Rehydrate the serialized RNG rather than any cached random_state, which
    # a consumer of the input may have advanced (PlayerHealthMutator did,
    # before it was fixed), so scoring and generation replay the recorded
    # transition.
    replay_state.__dict__.pop("random_state", None)
    next_state, _ = crafter_transition(
        replay_state, MAP_ACTION_TO_INDEX[cast(CrafterAction, action)]
    )
    return next_state


class RecordedDistractors:
    def __init__(self, generator):
        self.generator = generator
        self.hashes = []

    def __call__(self, transition, all_transitions, num_distractors):
        values = self.generator(transition, all_transitions, num_distractors)
        self.hashes.append(
            [
                sha256(
                    json.dumps(_gamestate_to_json(value), sort_keys=True).encode()
                ).hexdigest()
                for value in values
            ]
        )
        return values


def paired_context(shared, seed):
    context = copy.copy(shared)
    context.test_transitions = copy.deepcopy(shared.test_transitions)
    generator = copy.copy(shared.distractor_generator)
    if not isinstance(generator, CrafterDistractorGenerator):
        raise TypeError("Expected the existing Crafter distractor generator")
    generator.rng = random.Random(seed)
    context.distractor_generator = generator
    return context


def main(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda message: print(message, end="", flush=True), level="INFO")
    logger.add(output / "run.log", level="INFO")
    start = perf_counter()
    posterior_path = args.belief_dir / "fitted_belief_law.pkl"
    with posterior_path.open("rb") as file:
        posterior, adapter = cloudpickle.load(file)
    one = load_law_mixture(args.onelife_model)
    original = json.loads(args.onelife_results.read_text(encoding="utf-8"))
    config = original["evaluation_config"]
    trials = args.trials or config["num_trials"]
    samples = sorted(set(args.samples), reverse=True)
    if not samples or min(samples) < 1:
        raise ValueError("Predictive sample budgets must be positive")
    models = {
        "true_simulator": TrueTransitionWorldModel(
            true_simulator_step,
            lambda left, right: _gamestate_to_json(left) == _gamestate_to_json(right),
        ),
        "belief_law": CrafterBeliefWorldModel(posterior, adapter),
        "belief_law_map": CrafterBeliefWorldModel(posterior, adapter, map_only=True),
        "onelife": one,
    }
    extractor = one.observable_extractor
    if not isinstance(extractor, ObservableExtractor):
        raise TypeError(
            "Matched Crafter evaluation requires a Crafter observable extractor"
        )
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
    manifest = output / "predictive_bank_manifest.json"
    reuse = manifest.exists() and json.loads(manifest.read_text()) == fingerprint
    dump(manifest, fingerprint)
    # The new oracle has its own cache provenance; adding it does not invalidate
    # unchanged learned-model banks. Future simulator edits invalidate its bank.
    oracle_fingerprint = {
        **fingerprint,
        "oracle_source_sha256": {
            name: sha256(Path(name).read_bytes()).hexdigest()
            for name in [
                "runbook/06_matched_evaluation.py",
                "src/onelife/evaluator/crafter/utils.py",
                "external/crafter_oo/crafter_oo/functional_env.py",
                "external/crafter_oo/crafter_oo/state_reconstruction.py",
                "external/crafter_oo/crafter_oo/state_export.py",
                "external/crafter_oo/crafter_oo/engine.py",
                "external/crafter_oo/crafter_oo/objects.py",
                "external/crafter_oo/crafter_oo/constants.py",
            ]
        },
    }
    oracle_manifest = output / "true_simulator_bank_manifest.json"
    reuse_oracle = (
        reuse
        and oracle_manifest.exists()
        and json.loads(oracle_manifest.read_text()) == oracle_fingerprint
    )
    dump(oracle_manifest, oracle_fingerprint)
    all_results, diagnostics, timings = {}, {}, {}
    transition_hashes = []
    reference_distractors = None
    distractor_digest = None

    # Collect once, then copy the exact same transitions and distractor RNG for
    # each model/budget. Fresh factories need not serialize states identically.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    factory = CrafterEvaluationFactory(
        EnvConfig(size=adapter.template.size, view=adapter.template.view),
        policy_seed=config["eval_seed"],
    )
    shared_context = factory.create_context(
        EvaluationConfig(num_distractors=config["num_distractors"], num_trials=trials),
        config["num_transitions_per_scenario"],
    )
    shared_context.state_equivalence = lambda left, right: (
        extractor.get_observed_outcomes(left) == extractor.get_observed_outcomes(right)
    )
    if args.scenarios:
        shared_context.test_transitions = dict(
            list(shared_context.test_transitions.items())[: args.scenarios]
        )

    for name, model in models.items():
        tick = perf_counter()
        maximum = MatchedObservableModel(
            model,
            extractor,
            samples=max(samples),
            seed=args.seed,
            noise_logscore=args.noise_logscore,
        )
        bank_path = output / f"{name}_predictive_banks.json"
        model_reuse = reuse_oracle if name == "true_simulator" else reuse
        if model_reuse and bank_path.exists():
            maximum.load_banks(bank_path)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        context = paired_context(shared_context, config["eval_seed"])
        transitions = [t for ts in context.test_transitions.values() for t in ts]
        hashes = [maximum.key(t.prev_metadata, t.action) for t in transitions]
        if not transition_hashes:
            transition_hashes = hashes
        elif hashes != transition_hashes:
            raise RuntimeError("Models received different benchmark transitions")
        for index, transition in enumerate(transitions, start=1):
            maximum.predictive_statistics(transition.prev_metadata, transition.action)
            maximum.save_banks(bank_path)
            logger.info(
                "{} predictive bank {}/{} ({} samples)",
                name,
                index,
                len(transitions),
                max(samples),
            )
        timings[f"{name}_sampling_seconds"] = perf_counter() - tick
        for budget in samples:
            scorer = MatchedObservableModel(
                model,
                extractor,
                samples=budget,
                seed=args.seed,
                noise_logscore=args.noise_logscore,
                banks=maximum.banks,
            )
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            tick = perf_counter()
            eval_context = paired_context(shared_context, config["eval_seed"])
            recorder = RecordedDistractors(eval_context.distractor_generator)
            eval_context.distractor_generator = recorder
            result = evaluation_results_to_jsonable(
                Evaluator(eval_context).evaluate(scorer)
            )
            if reference_distractors is None:
                reference_distractors = recorder.hashes
                distractor_digest = sha256(
                    json.dumps(recorder.hashes).encode()
                ).hexdigest()
            elif recorder.hashes != reference_distractors:
                raise RuntimeError("Models or budgets received different distractors")
            key = f"{name}_n{budget}"
            all_results[key] = result
            diagnostics[key] = scorer.diagnostics()
            timings[f"{key}_evaluation_seconds"] = perf_counter() - tick
            dump(output / f"{key}_results.json", result)
            logger.info(
                "{} accuracy={:.4f} edit={:.4f} nonfinite={}",
                key,
                result["discriminative_accuracy"],
                result["edit_distance"]["raw"],
                diagnostics[key]["nonfinite_scores"],
            )

    assert reference_distractors is not None
    sensitivity = {
        name: {
            str(budget): all_results[f"{name}_n{budget}"]["discriminative_accuracy"]
            for budget in samples
        }
        for name in models
    }
    metadata = {
        **fingerprint,
        "num_scenarios": len(context.test_transitions),
        "num_transitions": len(transition_hashes),
        "num_trials": trials,
        "sample_budgets": samples,
        "accuracy_budget_sensitivity": sensitivity,
        "paired_transition_hashes": transition_hashes,
        "paired_distractor_sha256": distractor_digest,
        "paired_distractor_batches": len(reference_distractors),
        "observables": "saved OneLife ObservableExtractor with identical domain configuration",
        "candidate_equivalence": "equal saved OneLife observable projections; full-state equality remains the native evaluator default",
        "evaluator_sha256": sha256(
            Path("src/onelife/evaluator/core.py").read_bytes()
        ).hexdigest(),
        "scoring": "sum of smoothed posterior-predictive marginal log probabilities",
        "smoothing": "p(v)=(count(v)/N + exp(noise_logscore))/(1+D*exp(noise_logscore))",
        "domain_policy": "configured integer domain + OTHER; dynamic entity attributes + ABSENT; absent existence=0",
        "sampling": "fixed sample bank per state/action, identical budgets; separate isolated streams for ranking and generation",
        "distractor_pairing": "predictive sampling restores Python/NumPy/Torch/posterior RNG; identical evaluation seed and transition hashes",
        "heldout_usage": "evaluation only; no posterior update or candidate revision",
        "llm_network_calls": 0,
        "oracle": "actual Crafter transition from previous state/action; restores saved environment RNG; no held-out next-state access",
        "oracle_rng_policy": "discard copied random_state cache and rehydrate serialized environment RNG; distractor RNG mutations cannot change oracle replay",
        "oracle_source_sha256": oracle_fingerprint["oracle_source_sha256"],
        "timings": timings,
        "total_seconds": perf_counter() - start,
        "limits": [
            "Empirical marginal scores have Monte Carlo error; smaller nested budget checks sensitivity.",
            "Marginal scoring omits cross-observable correlations for every model; native BELIEF-LAW joint likelihood remains available separately.",
            "Additional scoring noise does not change predictive samples, fitting, or planning.",
            "Original saved OneLife accuracy uses a different scorer and is historical context only.",
            "Observable equivalence changes the evaluation target: candidates differing only in unscored fields count as correct for every model.",
            "OneLife's generator has its original internal noise; this is part of its predictive model, not the shared observation channel.",
            "True simulator has access to the complete previous state and saved environment RNG; it is a privileged reference ceiling.",
        ],
    }
    dump(
        output / "comparison.json",
        {
            "metadata": metadata,
            "results": all_results,
            "diagnostics": diagnostics,
            "historical_onelife": original["learned_world_model_performance"],
            "historical_true": original["true_world_model_performance"],
        },
    )
    rows = [
        {
            "model": name,
            "accuracy": r["discriminative_accuracy"],
            "recall": r["normalized_recall"],
            "reciprocal_rank": r["reciprocal_rank"],
            "edit_distance": r["edit_distance"]["raw"],
            "patch_iou": r["edit_distance"]["intersection_over_union"],
        }
        for name, r in all_results.items()
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_rows = []
    main_budget = max(samples)
    for scenario in all_results[f"belief_law_n{main_budget}"]["metrics_by_source"]:
        row = {"scenario": scenario}
        for name in models:
            mean = all_results[f"{name}_n{main_budget}"]["metrics_by_source"][scenario][
                "mean"
            ]
            row[f"{name}_accuracy"] = mean["discriminative_accuracy"]
        row["belief_minus_onelife"] = (
            row["belief_law_accuracy"] - row["onelife_accuracy"]
        )
        scenario_rows.append(row)
    with (output / "by_scenario.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(scenario_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_rows)
    report = [
        "# Matched observable evaluation: True simulator, BELIEF-LAW and OneLife",
        "",
        f"Frozen models; {metadata['num_scenarios']} scenarios, {len(transition_hashes)} transitions, {trials} trials, seed {args.seed}. Zero LLM calls.",
        f"Shared observable extractor, domains, additive noise logscore {args.noise_logscore}; same predictive sample budgets {samples}.",
        "Accuracy and rank recognize candidates with equal observable projections as equivalent.",
        "True simulator replays the actual transition from the previous state/action, including its saved environment RNG, through the same observable scorer. It never reads the held-out next state.",
        "Oracle replay rehydrates serialized RNG instead of the cached RNG advanced by PlayerHealthMutator during distractor generation.",
        "",
        "| Model / samples | Accuracy ↑ | Recall ↑ | Reciprocal rank ↑ | Edit distance ↓ | Patch IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['recall']:.4f} | {row['reciprocal_rank']:.4f} | {row['edit_distance']:.4f} | {row['patch_iou']:.4f} |"
        )
    report += [
        "",
        "All candidate log scores finite: "
        + str(all(d["nonfinite_scores"] == 0 for d in diagnostics.values())),
        f"Identical distractors verified over {len(reference_distractors)} batches for every model/budget.",
        "",
        "Generation uses the same isolated sample stream across scoring budgets. Different edit distances from the previous run reflect newly controlled sampling seeds, not model refitting.",
        "",
        "## Scope",
        "",
    ]
    report += ["- " + item for item in metadata["limits"]]
    report += [
        "",
        "## Historical scores",
        "",
        f"The original saved evaluation had True accuracy {original['true_world_model_performance']['discriminative_accuracy']:.2%} and learned OneLife accuracy {original['learned_world_model_performance']['discriminative_accuracy']:.2%}. The approximately 33% value was OneLife, not True. The historical scorer/equivalence differ from this matched observable benchmark.",
        "",
        "The original hard full-state BELIEF-LAW comparison remains in ../04_belief_law/. These scores replace that comparison for the matched observational benchmark.",
        "",
    ]
    (output / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("Saved matched comparison to {}", output / "comparison.md")


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
        "--output-dir", type=Path, default=Path("runbook_output/06_matched_evaluation")
    )
    parser.add_argument("--samples", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--noise-logscore", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=0)
    main(parser.parse_args())
