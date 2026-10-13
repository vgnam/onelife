# Runbook

The runbook runs the full OneLife pipeline. It has three steps, run in order. Each step reads the previous step's output, so running them in order with no arguments chains the whole pipeline.

Steps 1 and 2 call an LLM and need `GEMINI_API_KEY` in your `.env`. Step 3 does not. Steps 1 and 2 also take a `--debug` flag that runs a short version for a quick smoke test.

## 1. Generate an exploration trajectory

```bash
uv run --env-file .env python runbook/01_generate_exploration_trajectory.py
```

Runs one unguided exploration episode in Crafter and writes the trajectory under `runbook_output/01_exploration/`.

## 2. Synthesize laws

```bash
uv run --env-file .env python runbook/02_synthesize_laws.py
```

Reads the trajectory from step 1 and writes the synthesized laws under `runbook_output/02_synthesis/laws/`.

## 3. Optimize and evaluate

```bash
uv run --env-file .env python runbook/03_optimize_and_evaluate.py
```

Fits the world model from the laws and the trajectory, evaluates it, and writes `fitted_world_model.json` and `evaluation_results.json` under `runbook_output/03_optimize_and_evaluate/`.

## 4. BELIEF-LAW v2 baseline

```bash
uv run --env-file .env python runbook/04_belief_law.py
```

Reuses the GLM acquisition trajectory and the 50 programs in
`runbook_output/03_glm/fitted_world_model.json`; OneLife's fitted weights are
not used to fit the law posterior. Adds null and catch-all candidates, infers
trigger-aware co-write components, scores the acquisition history, and revises
only slots with catch-all mass above 0.5 (at most four LLM calls, using
`OPENAI_MODEL` and the configured OpenAI-compatible endpoint). Proposal responses
are cached under the output directory. `--revision-calls 0` runs the fixed-library
ablation. The component assignment budget defaults to 512; all truncations and
legacy-program errors are reported in `posterior_diagnostics.json`.

The world-model benchmark uses the saved OneLife evaluation configuration and
also reevaluates OneLife with matched scenario/distractor seeds. It writes
`comparison.md`, `comparison.json`, `comparison.csv`, and `by_scenario.csv` to
`runbook_output/04_belief_law/`. `--trials 1 --scenarios 2` is a smoke test.
Use `--onelife-model`, `--onelife-results`, and `--trajectory` to compare another
acquisition. The existing factory ignores its transition limit; actual scenario
and transition counts are recorded in the report.

The native engine supports correlated joint outcomes, Dirichlet learning of
identifiable stochastic outcomes, history-tree root sampling, PSRL and MAP
planner ablations, Tier 0 and Tier 1 reduction, and a closed-loop `BeliefLawAgent`.
Tier 2 is disabled. The existing Crafter benchmark measures one-step prediction,
so its results do not establish task-planning success. The legacy Crafter adapter
assumes independent marginals, uses conservative reads, and learns a portable
relative-edit catch-all dictionary; these limitations are explicit in the report.

`--preview-revision` creates all repair prompts locally and performs the full
evaluation without contacting an LLM. The supplied method specification is
saved in `docs/belief_law_v2.txt`. `runbook/05_audit_belief_law.py` diagnoses
held-out states outside predictive support after a run, without training on them.

```bash
uv run --env-file .env pytest tests/belief_law -q
```

## 6. Compare with shared observable scoring

```bash
uv run python runbook/06_matched_evaluation.py
```

Uses the frozen posterior from step 4, the saved GLM OneLife model, and the True
simulator reference without fitting or LLM calls. Every model supplies 128
predictive samples per state/action,
projected through the same saved OneLife observable extractor. A single scorer
sums marginal log probabilities with the same additive noise (`exp(-10)`) and
the same configured domains. Unseen values map to an OTHER category; dynamic
entity attributes also support ABSENT. The 64-sample nested bank checks Monte
Carlo sensitivity. This marginal benchmark omits joint correlations for both
models; the native BELIEF-LAW joint likelihood is retained in step 4. True replays
the actual Crafter transition using only the previous state and action. It restores
the saved environment RNG and is therefore a privileged reference ceiling.
Its predictions pass through the same observable projection and scorer.
Oracle replay discards the copied `random_state` cache and reloads the serialized
RNG. The health distractor mutator used to advance that cache on the benchmark
input (it no longer does), so results from runs before that fix are not
comparable. The oracle regression test covers replay after such mutations.

The benchmark collects one shared set of transitions and verifies identical
distractor hashes for every model and sample budget. Predictive RNG streams are
isolated from evaluator shuffling and distractor generation. The evaluator uses
observable equivalence: candidates with the same observable
projection count as correct, even if unscored fields differ. Full-state equality
remains the default for the existing native evaluator. Local sample banks
are cached with model/scorer fingerprints. `--samples 8 --trials 1 --scenarios 2`
is a smoke test. Results are saved separately under
`runbook_output/06_matched_evaluation/`; the previous scores are historical
context because they used different scorers. This is still the fixed-library
BELIEF-LAW ablation, not the LLM revision experiment.

## 7. Evaluate with the original full-state correctness criterion

```bash
uv run python runbook/07_strict_evaluation.py
```

Uses the same frozen BELIEF-LAW posterior, its MAP ablation, the saved OneLife
model, and True simulator. Accuracy and MRR use the original evaluator's
`WorldState` equality, with no observable-equivalence override. The native
likelihood run calls each model's original `evaluate_log_probability`. A separate
shared-likelihood control uses the verified 128-sample banks from step 6 while
keeping the original full-state correctness criterion.

Each model and scorer receives the same collected transitions and identical
distractors. The script also verifies that each model generates identical
predictions in the native and shared scoring runs. Sampling RNG streams are
isolated, and native scoring receives candidate-independent seeds. Negative
infinite native scores are retained; no noise is added to the native likelihood.
True retains the serialized-RNG replay repair. No fitting, posterior updates,
law revisions, or LLM calls occur. Results, diagnostics, and per-scenario metrics
are saved under `runbook_output/07_strict_evaluation/`.

Use `--scorers native` or `--scorers shared` for one scoring mode, and
`--scenarios 2 --trials 1 --output-dir runbook_output/07_strict_evaluation/smoke`
for a small check. Native likelihoods retain different predictive support across
methods; the shared control isolates the stricter correctness criterion without
mixing that change with a change of probability scorer.

## 8. Revise locally on train and evaluate the complete emitted state

```bash
uv run python runbook/08_repaired_evaluation.py
```

Fits a constant clock, a generic powered periodic daylight family, and guarded
joint survival programs using only the saved acquisition history. The revised
library is replayed on train. Catch-all tables back off to pooled joint train
signatures for unseen contexts; isolated numeric slots also have normalized
countable escape distributions with consistent sampling and analytic scoring.
Birth identities are portable, incomplete births are completed from train
prototypes, and chunk membership is rebuilt as structural bookkeeping.

True, original BELIEF-LAW/MAP, revised BELIEF-LAW/MAP, and frozen OneLife are
evaluated on identical inputs and distractors using the original `WorldState`
equality. All use the same 128-sample **full-state** measurement channel, whose
normalized typed noise prior supports unseen values. Unlike step 7's shared
control, daylight, clock and survival fields participate in ranking. Objects
remain joint tuple-valued observations; other fields use marginals.
View and chunks are excluded from ranking, matching the original equality,
and remain in the edit-distance diagnostics.
The True control must reach 100% accuracy. This channel is an approximation,
not the native joint law likelihood. Multi-field catch-alls still have finite
support; the synthesis family is deliberately limited.

No remote LLM calls occur. Original models/results are preserved. New fit,
hashes, diagnostics and comparison are saved separately under
`runbook_output/08_repaired_evaluation/`. `--fit-only` skips held-out evaluation.

## Running a step on its own

Each step takes the previous output as a path, so you can run a later step without running the earlier ones. The trajectory and laws from the paper are on HuggingFace at [OneLife-Crafter](https://huggingface.co/datasets/codezakh/OneLife-Crafter). Download them and pass them in. For example, to fit and evaluate from the published trajectory and laws:

```bash
hf download codezakh/OneLife-Crafter --repo-type dataset --local-dir OneLife-Crafter
uv run --env-file .env python runbook/03_optimize_and_evaluate.py \
    --trajectory OneLife-Crafter/trajectory/open_ended_run_00.jsonl \
    --laws-dir OneLife-Crafter
```
