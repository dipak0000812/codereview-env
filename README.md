---
title: Code Review Intelligence Environment
emoji: 📝
colorFrom: blue
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---
# CodeReview Intelligence - OpenEnv Environment

A Reinforcement Learning environment for simulating sequential decision-making in code review pipelines.

## Overview

This environment evaluates RL agents on multi-step code review tasks, requiring them to synthesize state observations (code diffs, dependency graphs, file history) to make risk-weighted decisions:
- **Assess risk level** (LOW / MEDIUM / HIGH / CRITICAL)
- **Compute blast radius** from dependency graphs
- **Identify optimal reviewers** based on context
- **Make merge decisions** (APPROVE / BLOCK / REQUEST_CHANGES)

The environment models real-world code review trade-offs between rapid merging and risk mitigation.

## Why Reinforcement Learning

Code reviews require sequential, state-dependent decisions. Early actions, such as initial risk classification, determine the required depth of subsequent analysis. The agent must balance execution speed (fast-tracking low-risk changes) against safety (performing deep context analysis for critical vulnerabilities). This environment cannot be solved as a single-step classification problem because the optimal trajectory depends entirely on the agent's prior actions and the evolving state observations.

## Decision Dynamics

Task 3 operates as a true Markov Decision Process (MDP) with variable trajectory lengths instead of fixed-step sequences:
- If the agent predicts LOW risk at the initial step, the episode may terminate immediately, yielding a higher reward for efficient decision-making.
- If this fast-track action is applied incorrectly to a critical vulnerability, the agent receives a severe penalty.
- Cautious actions reveal deeper context (dependency maps, file history), trading speed for higher decision accuracy.

## Episode Flow

1. **Initial Observation**: Agent observes the code diff and metadata.
2. **Risk Assessment**: Agent predicts the risk level (LOW/MEDIUM/HIGH/CRITICAL).
3. **Trajectory Branching**:
   - **Fast-Track**: If predicted risk is LOW, the episode terminates immediately for efficiency.
   - **Deep Analysis**: If predicted risk is higher, the environment reveals further context (blast radius, history).
4. **Final Reward**: A reward is assigned based on the optimal tradeoff between speed and review accuracy.

## Scenario Generation

The environment utilizes a procedural code review generator. Scenarios (including file diffs, dependency mappings, and repository history) are synthesized deterministically using pseudo-random seeds bound to the `episode_id`. This guarantees infinite state variance to prevent policy overfitting while maintaining 100% reproducibility for offline evaluation and training. This allows agents to learn generalizable policies rather than memorizing fixed scenarios.
## Quick Start
### Installation
```bash
pip install -r requirements.txt
```
### Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```
### Run Baseline Agent
```bash
python baseline.py --episodes 10
```
### Run Tests
```bash
pytest tests/ -v
```
## Three Tasks
### Task 1: Risk Classification (Easy)
Classify the risk level of a code change from the diff alone.
**Observation**: diff, filename, lines added/removed
**Action**: risk_level (LOW / MEDIUM / HIGH / CRITICAL)
**Grading**: Exact match = 1.0, 1 level = 0.5

### Task 2: Blast Radius Identification (Medium)
Identify all modules that would be affected by a change.

**Observation**: diff + dependency map (module  imports)
**Action**: affected_modules (list of module paths)
**Grading**: Jaccard similarity (clamped within valid scoring bounds).
### Task 3: Full Review Decision (Hard)
A multi-step sequential decision process where the trajectory dynamically branches based on the agent's actions.
**Observation**: Progressively reveals diff, dependency map, file history, and available reviewers across sequential steps.
**Action**: Sequential step execution (step 1: risk_level; step 2: affected_modules; step 3: reviewer, merge_decision).
**Grading**: Rewards are computed per-step and aggregated based on the trajectory taken by the agent.
- Early termination fast-tracking: High reward if optimal, severe penalty if risk was actually high or critical.
- Full review sequence: Composite weighted risk-assessment score.

**Safety Rule**: Strict mathematical penalty applied downstream to APPROVE actions on CRITICAL ground truths.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, start new episode |
| `/step` | POST | Take action, get reward |
| `/state` | GET | Get current state |
| `/tasks` | GET | List available tasks |
| `/grader` | POST | Grade arbitrary action |
| `/baseline` | GET | Get baseline scores |
| `/health` | GET | Health check |

## Architecture

```
+-------------------------------------------------------------+
|                    AGENT / BASELINE SCRIPT                  |
|                  CodeReviewEnv client                        |
+------------------------------------------------------------+
                       |
                       
+-------------------------------------------------------------+
|              DOCKER CONTAINER (port 7860)                    |
|                                                             |
|  +------------------------------------------------------+  |
|  |           FastAPI Server                              |  |
|  |                                                      |  |
|  |   /reset  environment.reset(task)                   |  |
|  |   /step   environment.step(action, episode_id)      |  |
|  |   /tasks  returns task list + action schema         |  |
|  |   /grader  grades arbitrary action vs scenario       |  |
|  |   /baseline  returns pre-computed scores            |  |
|  +------------------------------------------------------+  |
|                              |                               |
|         +----------------------------------------+         |
|                                                          |
|  +--------------+   +--------------+   +--------------+     |
|  | sessions.py  |   |  graders.py  |   |  dataset.py  |     |
|  | (episode     |   | (pure        |   | (loads JSON  |     |
|  |  store)      |   |  functions)  |   |  scenarios)  |     |
|  +--------------+   +--------------+   +--------------+     |
+-------------------------------------------------------------+
```

## Stateless Design

The environment is **stateless**. All episode state lives in `sessions.py` keyed by `episode_id`. This ensures:
- Multiple concurrent requests never contaminate each other
- Clean episode boundaries
- Thread-safe operation

## Baseline Scores

Baseline scores depend on procedurally generated scenarios and will vary across local and remote runs. The environment provides an endpoint (`/baseline`) to fetch expected statistical thresholds.

A trained agent's policy should significantly outperform naive sequential logic.

## Docker

```bash
# Build
docker build -t codereview-env .
# Run
docker run -p 7860:7860 codereview-env
# Test
docker run -p 7860:7860 codereview-env &
curl http://localhost:7860/health
```

## Project Structure

```
codereview-env/
├── app.py                 # FastAPI server
├── environment.py         # Stateless environment
├── graders.py             # Pure grading functions
├── sessions.py            # Episode session store
├── dataset.py             # JSON scenario loader
├── models.py              # Pydantic types
├── client.py              # Async HTTP client
├── baseline.py            # Deterministic baseline agent
├── inference.py           # Mandatory inference script
├── baseline_scores.json   # Pre-computed baseline scores
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── task1/             # 10 scenarios
│   ├── task2/             # 10 scenarios
│   └── task3/             # 10 scenarios
└── tests/
    ├── test_grader_variance.py
    └── test_environment.py
```

## Team ZerothLayer

- **Dipak Dhangar** - Architecture, Environment, Sessions
- **Tejas Patil** - Graders, Dataset, API

## License

MIT

<!-- force fresh validation -->
