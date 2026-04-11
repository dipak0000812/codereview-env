# CodeReview Intelligence Environment

A stateless Reinforcement Learning environment for training agents to perform deterministic, multi-step code review tasks — built for the Meta PyTorch OpenEnv Hackathon.

## Problem Statement

Code review requires synthesizing disparate signals: code diffs, dependency maps, file commit histories, and organizational context. Static analysis tools target syntactical errors but fail to capture architectural intent or blast radius. Evaluating agents on code review requires a reproducible, isolated environment that models the review process as a sequential decision problem rather than a single-turn text generation task.

## Why Reinforcement Learning

Code review is inherently sequential. An agent must assess risk, inspect dependency graphs, identify expert reviewers, and determine merge actions — without full context revealed upfront. Modeling this as a Markov Decision Process (MDP) allows agents to learn across multiple steps, where early misclassifications constrain accuracy on downstream decisions. This environment explicitly tests that progressive reasoning capacity.

## Task Overview

| Task | Name | Difficulty | Steps | Description |
|---|---|---|---|---|
| task1 | Risk Classification | Easy | 1 | Single-step risk level classification from code diff |
| task2 | Blast Radius Identification | Medium | 1 | Affected module identification via dependency graph traversal |
| task3 | Full Review Decision | Hard | 3 | Multi-step: risk → blast radius → reviewer + merge decision |

## Task Details

### Task 1: Risk Classification
Given a code diff (and full dependency map for context), predict the categorical risk level: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. Graded by ordinal distance with an additional safety penalty for severe underestimation (predicting `LOW` when truth is `CRITICAL`).

### Task 2: Blast Radius Identification
Given a code diff and a module dependency graph, output the list of directly impacted modules. Graded using Jaccard similarity with case-insensitive path matching to handle OS differences.

### Task 3: Full Review Decision (Multi-Step)
A progressive 3-step episode where context is revealed incrementally:
- **Step 1**: Given `diff` only → output `risk_level`
- **Step 2**: Given `diff + dependency_map` → output `affected_modules`
- **Step 3**: Given `diff + dependency_map + file_history + available_reviewers` → output `recommended_reviewer` and `merge_decision`

## Reward Design

All grader outputs are deterministic, pure functions returning floats in the strict open interval (0, 1).

| Component | Weight (task3) | Scoring Rule |
|---|---|---|
| Risk Classification | 25% | Ordinal distance table: {0→1.0, 1→0.5, 2→0.15, 3→0.0}. Extra 0.5× penalty for LOW vs CRITICAL. |
| Blast Radius | 30% | Jaccard similarity: \|A ∩ B\| / \|A ∪ B\|, case-insensitive |
| Reviewer | 20% | Exact match (case-insensitive, whitespace-stripped) |
| Merge Decision | 25% | Exact match. Safety rule: APPROVE on CRITICAL risk → −0.5 penalty |

Task 1 and Task 2 use only their respective components.

## Episode Flow (Task 3)

```
reset() → obs: {diff}
  ↓
step(risk_level) → reward₁ (25% weight), obs: {diff, dependency_map}
  ↓
step(affected_modules) → reward₂ (30% weight), obs: {diff, dependency_map, file_history, reviewers}
  ↓
step(reviewer + merge_decision) → reward₃ (45% weight), done=True
```

## Dataset

- **30 scenarios per task** (90 total) — diverse enterprise code review scenarios
- Domains: authentication, payments, ML pipelines, admin APIs, database migrations, search, notifications
- Risk distribution: LOW (20%), MEDIUM (30%), HIGH (30%), CRITICAL (20%)
- Blast radius: empty (40%), 1–2 modules (35%), 3+ modules (25%)

## Baseline Results

Deterministic naive agent (always predicts LOW, empty blast radius, first reviewer, APPROVE):

| Task | Baseline Score | Notes |
|---|---|---|
| task1 | 0.394 | Correct for LOW scenarios; wrong for HIGH/CRITICAL |
| task2 | ~0.001 | Empty list vs non-empty truth → near-zero Jaccard |
| task3 | 0.283 | Combines all three weaknesses |

## Environment Architecture

```
┌────────────────────────────────────────────────────┐
│                   Agent / Inference                │
└──────────────────────┬─────────────────────────────┘
                       │ HTTP/JSON
┌──────────────────────▼─────────────────────────────┐
│               FastAPI Server (server/app.py)        │
│  /reset  /step  /grader  /tasks  /info  /metrics   │
│                       │                            │
│  ┌─────────────┐  ┌───▼──────────┐                 │
│  │environment.py│  │  sessions.py │                 │
│  │ (MDP logic) │  │ (UUID store) │                 │
│  └──────┬──────┘  └──────────────┘                 │
│         │                                          │
│  ┌──────▼──────┐  ┌──────────────┐                 │
│  │  graders.py  │  │  dataset.py  │                 │
│  │ (pure funcs) │  │ (JSON files) │                 │
│  └─────────────┘  └──────────────┘                 │
└────────────────────────────────────────────────────┘
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Initialize episode, returns initial observation |
| POST | `/step` | Submit action, returns next observation + reward |
| POST | `/grader` | Grade a completed action payload |
| GET | `/tasks` | List tasks, action schema, step counts |
| GET | `/info` | Dataset stats, reward design documentation |
| GET | `/baseline` | Pre-computed naive baseline scores |
| GET | `/metrics` | Active session count + stale session purge |
| GET | `/health` | Server availability check |

## Running Locally

```bash
# Docker
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env

# Local dev
python -m venv venv
source venv/bin/activate          # Linux/macOS
# .\\venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
python -m pytest tests/ -v        # Run test suite (94 tests)
python baseline.py --episodes 10  # Run baseline agent
```

## Running Inference (LLM Agent)

```bash
export HF_TOKEN=hf_...
export ENV_URL=https://Dipak09-code-review-env.hf.space
python inference.py
```

The inference agent uses a chain-of-thought system prompt with step-specific structured prompts for each of the 3 tasks.

## Project Structure

```
codereview-env/
├── environment.py       # MDP: reset/step/grader with progressive context reveal
├── graders.py           # Pure scoring functions (deterministic, crash-safe)
├── sessions.py          # Thread-safe UUID session store (TTL-based cleanup)
├── dataset.py           # JSON scenario loader for 30×3 scenarios
├── models.py            # Pydantic schemas: Action, Observation, State
├── inference.py         # LLM agent with CoT prompts and robust output parsing
├── baseline.py          # Naive baseline agent for comparison
├── server/app.py        # FastAPI routes: reset, step, grader, info, metrics
├── data/
│   ├── task1/           # 20 scenarios: risk classification
│   ├── task2/           # 20 scenarios: blast radius identification
│   └── task3/           # 20 scenarios: full multi-step review
├── tests/
│   ├── test_environment.py      # 35 environment lifecycle tests
│   ├── test_grader_variance.py  # 59 grader correctness + variance tests
│   └── test_clamp_compliance.py # Boundary compliance tests
├── openenv.yaml         # spec_version: 1, type: space, runtime: fastapi
└── Dockerfile           # Container definition
```

## Design Decisions

**Progressive Context Reveal (Task 3)**: The environment deliberately withholds `dependency_map`, `file_history`, and `available_reviewers` at step 1, revealing them progressively. This forces agents to reason about risk before seeing the full picture — matching real review workflows where risk judgment precedes dependency analysis.

**Thread-Safe Session Store**: All episode state lives in `sessions.py` (a UUID-keyed dict protected by `threading.Lock`). The environment class holds no state. Sessions are deleted immediately after episode completion and purged after 30 minutes of inactivity.

**Deterministic Grading**: All grader functions are pure (same input → same output, no external dependencies). The Jaccard and risk functions use ordinal distance and set-theory operations — no ML models, no heuristics that could drift.

**Safety Penalty**: APPROVE on CRITICAL risk applies a −0.5 penalty to merge score regardless of correctness. This discourages agents from rubber-stamping high-severity changes and reflects real-world review policy.

**Bounded Outputs**: `_clamp()` ensures all grader returns are strictly in (0, 1), never exactly 0.0 or 1.0, satisfying the OpenEnv validator's strict interval requirement.

## Limitations

- Action spaces use fixed vocabulary (no free-text justification)
- Blast radius is based on static dependency maps (no runtime analysis)
- Reviewer assignment is based on name strings (no role/expertise modeling)

## License

MIT
