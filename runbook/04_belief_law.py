"""Fit BELIEF-LAW v2 on the existing acquisition and compare with saved OneLife.

uv run --env-file .env python runbook/04_belief_law.py
"""

import csv
import json
import random
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cloudpickle
import numpy as np
import torch
from crafter_oo.functional_env import EnvConfig
from crafter_oo.state_export import WorldState
from loguru import logger

from onelife.balrog_evaluator import TrajectoryStep
from onelife.belief_law.core import co_write_components
from onelife.belief_law.crafter import (
    CrafterAdapter,
    CrafterBeliefWorldModel,
    build_library,
)
from onelife.belief_law.revision import LibraryReviser
from onelife.evaluator import EvaluationConfig, Evaluator
from onelife.evaluator.crafter.factory import CrafterEvaluationFactory
from onelife.evaluator.serde import evaluation_results_to_jsonable
from onelife.our_method.crafter.serde import load_law_mixture

METRICS = ["discriminative_accuracy", "normalized_recall", "reciprocal_rank"]


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda message: print(message, end="", flush=True), level="INFO")
    logger.add(output / "run.log", level="INFO")
    start = perf_counter()
    one = load_law_mixture(args.onelife_model)
    steps = [
        TrajectoryStep[WorldState].model_validate_json(line)
        for line in args.trajectory.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    adapter = CrafterAdapter(steps[0].info, args.outcome_budget)
    flat = [adapter.encode(s.info) for s in steps]
    for encoded, step in zip(flat, steps):
        adapter.remember(encoded, step.info)
    history = [(flat[i - 1], steps[i].action, flat[i]) for i in range(1, len(steps))]
    logger.info(
        "Building slots from {} acquired laws and {} transitions",
        len(one.laws),
        len(history),
    )
    posterior = build_library(
        adapter,
        [w.law for w in one.laws],
        history,
        beam=args.beam,
        length_penalty=args.length_penalty,
        seed=args.seed,
    )
    dump(output / "initial_library.json", posterior.diagnostics())
    logger.info("Initial components: {}", posterior.diagnostics()["component_sizes"])
    if args.prepare_only:
        return
    fit_start = perf_counter()
    fit_cache = output / "acquisition_posterior.pkl"
    fingerprint = {
        "trajectory": sha256(args.trajectory.read_bytes()).hexdigest(),
        "laws": sha256(args.onelife_model.read_bytes()).hexdigest(),
        "beam": args.beam,
        "length_penalty": args.length_penalty,
        "outcome_budget": args.outcome_budget,
        "implementation": {
            name: sha256(
                (Path("src/onelife/belief_law") / name).read_bytes()
            ).hexdigest()
            for name in ["core.py", "crafter.py"]
        },
    }
    manifest = output / "acquisition_cache.json"
    if (
        fit_cache.exists()
        and manifest.exists()
        and (
            json.loads(manifest.read_text()) == fingerprint
            or (
                args.reuse_acquisition_cache
                and {
                    k: v
                    for k, v in json.loads(manifest.read_text()).items()
                    if k != "implementation"
                }
                == {k: v for k, v in fingerprint.items() if k != "implementation"}
            )
        )
    ):
        with fit_cache.open("rb") as file:
            posterior, adapter, fit_seconds = cloudpickle.load(file)
        logger.info(
            "Loaded acquisition checkpoint; verified inputs, evaluation-repair override={}",
            args.reuse_acquisition_cache,
        )
    else:
        logger.info("Fitting factored posterior")
        posterior.fit(
            history,
            progress=lambda step, total: (
                logger.info("Posterior update {}/{}", step, total)
                if step % 25 == 0 or step == total
                else None
            ),
        )
        fit_seconds = perf_counter() - fit_start
        with fit_cache.open("wb") as file:
            cloudpickle.dump((posterior, adapter, fit_seconds), file)
        dump(manifest, fingerprint)
    logger.info("Posterior fitted in {:.2f}s", fit_seconds)
    dump(output / "posterior_before_revision.json", posterior.diagnostics())
    reviser = LibraryReviser(
        adapter,
        output / "proposals",
        threshold=args.catchall_threshold,
        max_calls=args.revision_calls,
    )
    if args.revision_calls:
        logger.info(
            "Revising only slots above catch-all mass {} (cap {} calls)",
            args.catchall_threshold,
            args.revision_calls,
        )
        reviser.run(posterior, preview=args.preview_revision)
    with (output / "fitted_belief_law.pkl").open("wb") as file:
        cloudpickle.dump((posterior, adapter), file)
    diagnostics = posterior.diagnostics()
    diagnostics["plain_overlap_component_slot_counts"] = [
        len(c) for c in co_write_components(posterior.slots, False)
    ]
    diagnostics["trigger_aware_component_slot_counts"] = [
        len(c.slots) for c in posterior.components
    ]
    diagnostics["legacy_adapter_approximations"] = dict(adapter.approximations)
    diagnostics["legacy_program_errors"] = dict(adapter.failures)
    diagnostics["revision_events"] = reviser.events
    diagnostics["current_beam_components"] = sum(
        c.full_size > len(c.assignments) for c in posterior.components
    )
    diagnostics["shared_catchall_count_components"] = sum(
        sum(i in posterior.catchalls for i in c.slots) > 1 for c in posterior.components
    )
    diagnostics["limitations"] = [
        "Legacy marginal laws assume independent outcome fields; original cross-field correlation is unavailable.",
        "Opaque Python laws use conservative full-state reads, so Tier 0 cannot promise a speed gain here.",
        "Write and action contracts were inferred on acquisition probes; unseen writes are rejected at runtime.",
        "Catch-all support is an empirical dictionary of relative edits, frozen before prequential scoring.",
        "Co-write catch-all tables share observed-projection counts; latent conditional counts are approximated.",
        "The Crafter decoder drops incomplete sampled entity births and logs this projection; it does not infer missing attributes.",
        "BELIEF-LAW scores full encoded states; OneLife scores a subset with expanded-support noise and soft persistence penalties, so ranking also reflects scoring support.",
        "Legacy stochastic probabilities remain fixed; native joint candidates support Dirichlet learning.",
        "Beam retains highest-prior assignments before scoring, with null and catch-all escape assignments; exactness is lost when used.",
        "No task-success claim follows from the one-step world-model benchmark; planning requires its own experiment.",
        "Tier 2 is disabled, as permitted by the v2 core specification.",
    ]
    dump(output / "posterior_diagnostics.json", diagnostics)
    dump(
        output / "library.json",
        [
            {
                "name": s.name,
                "writes": sorted(s.writes),
                "triggers": sorted(s.triggers) if s.triggers else None,
                "candidates": [
                    {
                        "name": c.name,
                        "kind": c.kind,
                        "length": c.length,
                        "source": c.source,
                    }
                    for c in s.candidates
                ],
            }
            for s in posterior.slots
        ],
    )
    saved = json.loads(args.onelife_results.read_text(encoding="utf-8"))
    config = saved["evaluation_config"]
    num_trials = args.trials or config["num_trials"]
    models = {
        "belief_law": CrafterBeliefWorldModel(posterior, adapter),
        "belief_law_map": CrafterBeliefWorldModel(posterior, adapter, map_only=True),
        "onelife_paired": one,
    }
    results, evaluation_seconds = {}, {}
    for name, model in models.items():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        posterior.rng = np.random.default_rng(args.seed)
        factory = CrafterEvaluationFactory(
            EnvConfig(size=steps[0].info.size, view=steps[0].info.view),
            policy_seed=config["eval_seed"],
        )
        context = factory.create_context(
            EvaluationConfig(
                num_distractors=config["num_distractors"], num_trials=num_trials
            ),
            num_transitions_per_scenario=config["num_transitions_per_scenario"],
        )
        if args.scenarios:
            context.test_transitions = dict(
                list(context.test_transitions.items())[: args.scenarios]
            )
        logger.info(
            "Evaluating {} on {} scenarios / {} transitions / {} trials",
            name,
            len(context.test_transitions),
            sum(map(len, context.test_transitions.values())),
            num_trials,
        )
        tick = perf_counter()
        results[name] = evaluation_results_to_jsonable(
            Evaluator(context).evaluate(model)
        )
        evaluation_seconds[name] = perf_counter() - tick
        dump(output / f"{name}_results.json", results[name])
        logger.info(
            "{} accuracy={:.4f} edit={:.4f} elapsed={:.2f}s",
            name,
            results[name]["discriminative_accuracy"],
            results[name]["edit_distance"]["raw"],
            evaluation_seconds[name],
        )
    diagnostics["legacy_adapter_approximations"] = dict(adapter.approximations)
    diagnostics["legacy_program_errors"] = dict(adapter.failures)
    diagnostics["adapter_counter_scope"] = "acquisition, revision and evaluation"
    dump(output / "posterior_diagnostics.json", diagnostics)
    results["onelife_saved"] = saved["learned_world_model_performance"]
    metadata = {
        "trajectory": str(args.trajectory.resolve()),
        "onelife_model": str(args.onelife_model.resolve()),
        "onelife_saved_results": str(args.onelife_results.resolve()),
        "input_sha256": {
            str(p): sha256(p.read_bytes()).hexdigest()
            for p in [args.trajectory, args.onelife_model, args.onelife_results]
        },
        "train_transitions": len(history),
        "seed": args.seed,
        "evaluation_config": {**config, "num_trials": num_trials},
        "beam": args.beam,
        "outcome_budget": args.outcome_budget,
        "length_penalty": args.length_penalty,
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": perf_counter() - start,
        "pairing": "fresh deterministic contexts and distractor seeds per model; original saved run has uncontrolled sampling RNG",
        "factory_transition_limit": "factory caps each scenario at num_transitions_per_scenario; actual counts reported",
        "revision_calls": reviser.network_calls,
        "revision_windows": reviser.calls,
        "revision_preview": args.preview_revision,
        "run_mode": "fixed_library"
        if reviser.network_calls == 0
        else "bounded_llm_revision",
    }
    dump(
        output / "comparison.json",
        {"metadata": metadata, "results": results, "posterior": diagnostics},
    )
    rows = []
    for name, perf in results.items():
        rows.append(
            {
                "model": name,
                **{k: perf[k] for k in METRICS},
                "edit_raw": perf["edit_distance"]["raw"],
                "edit_normalized": perf["edit_distance"]["normalized"],
                "edit_iou": perf["edit_distance"]["intersection_over_union"],
            }
        )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_rows = []
    for scenario, perf in results["belief_law"]["metrics_by_source"].items():
        b = perf["mean"]
        o = results["onelife_paired"]["metrics_by_source"][scenario]["mean"]
        scenario_rows.append(
            {
                "scenario": scenario,
                "belief_accuracy": b["discriminative_accuracy"],
                "onelife_accuracy": o["discriminative_accuracy"],
                "accuracy_delta": b["discriminative_accuracy"]
                - o["discriminative_accuracy"],
                "belief_edit": b["edit_distance"]["raw"],
                "onelife_edit": o["edit_distance"]["raw"],
            }
        )
    with (output / "by_scenario.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(scenario_rows[0]))
        writer.writeheader()
        writer.writerows(scenario_rows)
    wins = sum(r["accuracy_delta"] > 1e-12 for r in scenario_rows)
    losses = sum(r["accuracy_delta"] < -1e-12 for r in scenario_rows)
    report = [
        "# BELIEF-LAW v2 vs OneLife (GLM acquisition)",
        "",
        f"Same {len(history)} acquisition transitions; {num_trials} evaluation trials, seed {args.seed}.",
        f"LLM revision: {reviser.network_calls} network calls, {reviser.calls} windows; preview={args.preview_revision}.",
        "Fixed-library evaluation; LLM revision has not run."
        if reviser.network_calls == 0
        else "Evaluation after bounded LLM library revision.",
        "",
        "| Model | Accuracy ↑ | Recall ↑ | Reciprocal rank ↑ | Edit distance ↓ | Patch IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['discriminative_accuracy']:.4f} | {row['normalized_recall']:.4f} | {row['reciprocal_rank']:.4f} | {row['edit_raw']:.4f} | {row['edit_iou']:.4f} |"
        )
    report += [
        "",
        f"Accuracy vs paired OneLife: {wins} wins, {losses} losses, {len(scenario_rows) - wins - losses} ties across {len(scenario_rows)} scenarios.",
        "",
        f"Posterior: {len(posterior.slots)} slots, {len(posterior.components)} components; {sum(len(c.assignments) for c in posterior.components)} retained assignments.",
        f"Fit: {fit_seconds:.2f}s; evaluation times: {evaluation_seconds}.",
        "",
        "## Scope and limitations",
        "",
    ]
    report += ["- " + item for item in diagnostics["limitations"]]
    report += [
        "",
        "Approximations: " + json.dumps(diagnostics["approximations"]),
        "Legacy adapter: " + json.dumps(diagnostics["legacy_adapter_approximations"]),
        "",
        "Stored vs paired OneLife differs because the original run did not seed all sampling RNGs.",
        "The paired MAP ablation fixes identities for both generation and ranking.",
        "Results describe this acquired library and one evaluation seed, not a general superiority claim.",
    ]
    (output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("Saved comparison to {}", output / "comparison.md")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path(
            "runbook_output/01_exploration/crafter/open_ended/open_ended_run_00.jsonl"
        ),
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
        "--output-dir", type=Path, default=Path("runbook_output/04_belief_law")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--outcome-budget", type=int, default=128)
    parser.add_argument("--length-penalty", type=float, default=0.01)
    parser.add_argument("--catchall-threshold", type=float, default=0.5)
    parser.add_argument("--revision-calls", type=int, default=4)
    parser.add_argument(
        "--preview-revision",
        action="store_true",
        help="Write proposed LLM prompts locally without sending them",
    )
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--scenarios", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--reuse-acquisition-cache",
        action="store_true",
        help="Reuse this run's acquisition checkpoint after a verified evaluation-only adapter repair",
    )
    main(parser.parse_args())
