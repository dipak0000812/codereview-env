# CodeReview Intelligence - OpenEnv Environment

An RL environment for training agents to perform intelligent code review, inspired by the PRISM-AI project from the GitLab AI Hackathon.

## Overview

This environment trains RL agents to review code the way senior engineers do:
- **Assess risk level** (LOW / MEDIUM / HIGH / CRITICAL)
- **Compute blast radius** from dependency graphs
- **Identify optimal reviewers** based on context
- **Make merge decisions** (APPROVE / BLOCK / REQUEST_CHANGES)

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
**Grading**: Exact match = 1.0, ±1 level = 0.5

### Task 2: Blast Radius Identification (Medium)
Identify all modules that would be affected by a change.

**Observation**: diff + dependency map (module → imports)
**Action**: affected_modules (list of module paths)
**Grading**: Jaccard similarity |predicted ∩ truth| / |predicted ∪ truth|

### Task 3: Full Review Decision (Hard)
Make a complete review decision synthesizing all available context.

**Observation**: diff + dependency map + file history + available reviewers
**Action**: All 4 fields (risk_level, affected_modules, reviewer, merge_decision)
**Grading**: Composite weighted score
- Risk level: 25%
- Blast radius: 30%
- Reviewer: 20%
- Merge decision: 25%

**Safety Rule**: APPROVE on CRITICAL risk → -0.5 penalty

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
┌─────────────────────────────────────────────────────────────┐
│                    AGENT / BASELINE SCRIPT                  │
│                  CodeReviewEnv client                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              DOCKER CONTAINER (port 7860)                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           FastAPI Server                              │  │
│  │                                                      │  │
│  │   /reset → environment.reset(task)                   │  │
│  │   /step  → environment.step(action, episode_id)      │  │
│  │   /tasks → returns task list + action schema         │  │
│  │   /grader → grades arbitrary action vs scenario       │  │
│  │   /baseline → returns pre-computed scores            │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│         ┌────────────────────┼────────────────────┐         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐     │
│  │ sessions.py  │   │  graders.py  │   │  dataset.py  │     │
│  │ (episode     │   │ (pure        │   │ (loads JSON  │     │
│  │  store)      │   │  functions)  │   │  scenarios)  │     │
│  └──────────────┘   └──────────────┘   └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Stateless Design

The environment is **stateless**. All episode state lives in `sessions.py` keyed by `episode_id`. This ensures:
- Multiple concurrent requests never contaminate each other
- Clean episode boundaries
- Thread-safe operation

## Baseline Scores

| Task | Baseline Score | Description |
|------|----------------|-------------|
| task1 | 0.394 | Naive risk classification (Always LOW) |
| task2 | 0.000 | Naive module selection (Always empty) |
| task3 | 0.283 | Naive sequential full decision (3 steps, weighted) |

A trained agent should significantly outperform these baselines.

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
├── server/
│   ├── __init__.py
│   ├── app.py          # FastAPI application
│   ├── environment.py  # Stateless environment
│   ├── graders.py      # Pure grading functions
│   ├── dataset.py      # Scenario loader
│   └── sessions.py     # Episode store
├── data/
│   ├── task1/          # 10 risk classification scenarios
│   ├── task2/          # 10 blast radius scenarios
│   └── task3/          # 10 full review scenarios
├── tests/
│   ├── test_grader_variance.py
│   └── test_environment.py
├── models.py           # Pydantic types
├── client.py           # HTTP client
├── baseline.py         # Deterministic baseline agent
├── baseline_scores.json
├── openenv.yaml        # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Team ZerothLayer

- **Dipak Dhangar** - Architecture, Environment, Sessions
- **Tejas Patil** - Graders, Dataset, API

## License

MIT
