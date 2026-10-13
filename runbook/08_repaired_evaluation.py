"""Train-only local repair, then paired original-criterion full-state evaluation.

uv run python runbook/08_repaired_evaluation.py
Historical artifacts are read-only. No remote LLM or network calls are made.
"""

import copy
import csv
import json
import os
import runpy
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cloudpickle
from crafter_oo.functional_env import EnvConfig
from loguru import logger

from onelife.belief_law.crafter import CrafterBeliefWorldModel
from onelife.belief_law.full_evaluation import FullStateScorer
from onelife.belief_law.repair import revise_locally
from onelife.evaluator import EvaluationConfig, Evaluator
from onelife.evaluator.baselines import TrueTransitionWorldModel
from onelife.evaluator.crafter.components import _gamestate_to_json
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.serde import evaluation_results_to_jsonable
from onelife.our_method.crafter.serde import load_law_mixture


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda message: print(message, end="", flush=True), level="INFO")
    logger.add(output / "run.log", level="INFO")
    helpers = runpy.run_path("runbook/06_matched_evaluation.py")
    strict = runpy.run_path("runbook/07_strict_evaluation.py")
    original = Path("runbook_output/04_belief_law/fitted_belief_law.pkl")
    one_path = Path("runbook_output/03_glm/fitted_world_model.json")
    source_paths = [
        "src/onelife/belief_law/" + name
        for name in (
            "core.py",
            "crafter.py",
            "repair.py",
            "full_evaluation.py",
            "evaluation.py",
            # Swapped in for FullStateScorer by cached evaluation drivers.
            "predictive_cache.py",
            "onelife_predictive_cache.py",
        )
    ]
    fingerprint = {
        "original_posterior": file_hash(original),
        "onelife_model": file_hash(one_path),
        "implementation": {path: file_hash(path) for path in source_paths},
    }
    fit_fingerprint = {
        **fingerprint,
        "implementation": {
            p: h
            for p, h in fingerprint["implementation"].items()
            if Path(p).name in {"core.py", "crafter.py", "repair.py"}
        },
    }
    with original.open("rb") as file:
        old_posterior, old_adapter = cloudpickle.load(file)
    fit_dir = args.fit_dir or output
    fitted = fit_dir / "fitted_belief_law.pkl"
    manifest = fit_dir / "repair_manifest.json"
    if (
        fitted.exists()
        and manifest.exists()
        and {
            **json.loads(manifest.read_text()),
            "implementation": {
                p: h
                for p, h in json.loads(manifest.read_text())["implementation"].items()
                if Path(p).name in {"core.py", "crafter.py", "repair.py"}
            },
        }
        == fit_fingerprint
    ):
        with fitted.open("rb") as file:
            posterior, adapter = cloudpickle.load(file)
        logger.info("Loaded verified train-only repair checkpoint")
    else:
        logger.info(
            "Inducing local revisions on {} train transitions",
            len(old_posterior.history),
        )
        start = perf_counter()
        posterior, adapter, history, report = revise_locally(
            copy.deepcopy(old_posterior), old_adapter
        )
        logger.info("Train daylight fit: {}", report["clock"])
        dump(output / "local_revision.json", report)
        logger.info(
            "Replaying full train posterior with {} proposals", len(report["proposals"])
        )
        posterior.fit(
            history,
            progress=lambda step, total: (
                logger.info("Train replay {}/{}", step, total)
                if step % 25 == 0 or step == total
                else None
            ),
        )
        adapter._decoded.clear()
        with fitted.open("wb") as file:
            cloudpickle.dump((posterior, adapter), file)
        dump(manifest, fit_fingerprint)
        dump(output / "posterior_diagnostics.json", posterior.diagnostics())
        report["fit_seconds"] = perf_counter() - start
        dump(output / "local_revision.json", report)
    if args.fit_only:
        return
    one = load_law_mixture(one_path)
    historical = json.loads(
        Path("runbook_output/03_glm/evaluation_results.json").read_text()
    )
    config = historical["evaluation_config"]
    strict["seed_rng"](args.seed)
    shared = (
        cloudpickle.loads(args.context_path.read_bytes())
        if getattr(args, "context_path", None)
        else CrafterEvaluationFactory(
            EnvConfig(size=adapter.template.size, view=adapter.template.view),
            policy_seed=config["eval_seed"],
        ).create_context(
            EvaluationConfig(
                num_distractors=config["num_distractors"],
                num_trials=args.trials or config["num_trials"],
            ),
            config["num_transitions_per_scenario"],
        )
    )
    shared.state_equivalence = None
    if args.scenarios:
        shared.test_transitions = dict(
            list(shared.test_transitions.items())[: args.scenarios]
        )
    models = {
        "true_simulator": TrueTransitionWorldModel(
            helpers["true_simulator_step"],
            lambda l, r: _gamestate_to_json(l) == _gamestate_to_json(r),
        ),
        "belief_law_before": CrafterBeliefWorldModel(old_posterior, old_adapter),
        "map_before": CrafterBeliefWorldModel(
            old_posterior, old_adapter, map_only=True
        ),
        "belief_law_repaired": CrafterBeliefWorldModel(posterior, adapter),
        "map_repaired": CrafterBeliefWorldModel(posterior, adapter, map_only=True),
        "onelife": one,
    }
    if args.models:
        models = {name: model for name, model in models.items() if name in args.models}
    results, diagnostics, timings = {}, {}, {}
    reference = None
    for name, model in models.items():
        scorer = FullStateScorer(model, samples=args.samples, seed=args.seed)
        strict["seed_rng"](args.seed)
        context = helpers["paired_context"](shared, config["eval_seed"])

        class TracedDistractors(helpers["RecordedDistractors"]):
            def __call__(self, *args, **kwargs):
                generated = super().__call__(*args, **kwargs)
                if len(self.hashes) == 1:
                    dump(
                        output / "first_distractor_batch.json", {"hash": self.hashes[0]}
                    )
                return generated

        distractors = TracedDistractors(context.distractor_generator)
        context.distractor_generator = distractors
        start = perf_counter()
        logger.info("Evaluating {}: full-state score, original equality", name)
        result = evaluation_results_to_jsonable(Evaluator(context).evaluate(scorer))
        if reference is None:
            reference = distractors.hashes
        elif reference != distractors.hashes:
            raise RuntimeError("Compared models received different distractors")
        results[name], diagnostics[name] = result, scorer.diagnostics()
        timings[name] = perf_counter() - start
        dump(output / (name + "_results.json"), result)
        logger.info(
            "{} accuracy={:.2%} MRR={:.4f} edits={:.3f} IoU={:.4f}",
            name,
            result["discriminative_accuracy"],
            result["reciprocal_rank"],
            result["edit_distance"]["raw"],
            result["edit_distance"]["intersection_over_union"],
        )
        if name == "true_simulator" and result["discriminative_accuracy"] != 1:
            raise RuntimeError("Full-state oracle control failed")
    if file_hash(original) != fingerprint["original_posterior"]:
        raise RuntimeError("Historical posterior changed during repair")
    if {path: file_hash(path) for path in source_paths} != fingerprint[
        "implementation"
    ]:
        raise RuntimeError("Implementation sources changed during the run")
    assert reference is not None
    metadata = {
        **fingerprint,
        "seed": args.seed,
        "samples": args.samples,
        "trials": context.config.num_trials,
        "scenarios": len(shared.test_transitions),
        "test_transitions": sum(map(len, shared.test_transitions.values())),
        "train_transitions": len(posterior.history),
        "network_calls": 0,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "frozen_context_sha256": file_hash(args.context_path)
        if getattr(args, "context_path", None)
        else None,
        "heldout_usage": "evaluation only",
        "identical_distractor_batches": len(reference),
        "distractor_batch_sha256": sha256(json.dumps(reference).encode()).hexdigest(),
        "input_transition_sha256": sha256(
            json.dumps(
                [
                    [
                        _gamestate_to_json(t.prev_metadata),
                        t.action,
                        _gamestate_to_json(t.next_metadata),
                    ]
                    for transitions in shared.test_transitions.values()
                    for t in transitions
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "timings": timings,
        "limits": [
            "All six models use the same full-state MC marginal channel and original WorldState equality. These scores differ from historical native-likelihood scores.",
            "Numeric isolated catch-alls have an analytic, normalized discrete escape; multi-field catch-alls back off to pooled joint train signatures and remain finite-support.",
            "Survival revision uses a finite guarded-program family fitted on train, with posterior replay; it is not unrestricted program synthesis.",
            "Birth completion uses train prototypes. Structural decoding is a declared projection; ranking scores the emitted states, not an incorrect symbolic likelihood of pre-projection states.",
            "Objects are joint tuple observations; other scored fields use marginals. View and chunks are excluded from ranking exactly as in original WorldState equality, and remain in edit-distance diagnostics. Cross-field correlations and exact posterior joint likelihood are not preserved by the shared channel.",
            "Periodic extrapolation is inferred from one acquisition trajectory and is not guaranteed outside its coverage.",
        ],
    }
    dump(
        output / "comparison.json",
        {"metadata": metadata, "results": results, "diagnostics": diagnostics},
    )
    rows = [
        {
            "model": name,
            "accuracy": r["discriminative_accuracy"],
            "mrr": r["reciprocal_rank"],
            "edit_distance": r["edit_distance"]["raw"],
            "patch_iou": r["edit_distance"]["intersection_over_union"],
        }
        for name, r in results.items()
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Train-only repair: original-criterion paired comparison",
        "",
        f"{metadata['train_transitions']} train transitions; {metadata['test_transitions']} test transitions, {metadata['trials']} trials, {args.samples} samples, seed {args.seed}.",
        "Full-state predictive scoring channel and original WorldState equality for every model. No network calls.",
        "",
        "| Model | Accuracy ↑ | MRR ↑ | Edit distance ↓ | Patch IoU ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['accuracy']:.2%} | {row['mrr']:.4f} | {row['edit_distance']:.4f} | {row['patch_iou']:.4f} |"
        )
    report += ["", "## Limits", ""] + ["- " + text for text in metadata["limits"]]
    (output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("Saved {}", output / "comparison.md")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runbook_output/08_repaired_evaluation")
    )
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--context-path", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "true_simulator",
            "belief_law_before",
            "map_before",
            "belief_law_repaired",
            "map_repaired",
            "onelife",
        ],
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
