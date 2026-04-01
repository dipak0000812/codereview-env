"""FastAPI server for CodeReview environment.

Uses OpenEnv's official create_app() factory:
    from openenv.core.env_server.http_server import create_app

Custom endpoints added on top:
    GET  /tasks    — task list + action schema
    POST /grader   — grade arbitrary action vs scenario
    GET  /baseline — pre-computed baseline scores
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

try:
    from models import CodeReviewAction, CodeReviewObservation
    from server.environment import CodeReviewEnvironment
except ImportError:
    from ..models import CodeReviewAction, CodeReviewObservation
    from .environment import CodeReviewEnvironment


# ── Create app using official OpenEnv factory ────────────────────────────────
# create_app takes the Environment CLASS (not instance) as a factory callable
# This ensures each WebSocket session gets its own fresh instance

app = create_app(
    env=CodeReviewEnvironment,           # Class used as factory
    action_cls=CodeReviewAction,
    observation_cls=CodeReviewObservation,
    env_name="codereview-env",
    max_concurrent_envs=100,
)


# ── Custom endpoints added on top ─────────────────────────────────────────────

@app.get("/tasks")
async def get_tasks():
    """Return available tasks and action schema."""
    return {
        "tasks": [
            {
                "name": "task1",
                "description": "Classify risk level from code diff",
                "difficulty": "easy",
                "action_fields": ["episode_id", "risk_level"]
            },
            {
                "name": "task2",
                "description": "Identify blast radius from dependency map",
                "difficulty": "medium",
                "action_fields": ["episode_id", "affected_modules"]
            },
            {
                "name": "task3",
                "description": "Full review decision with composite scoring",
                "difficulty": "hard",
                "action_fields": [
                    "episode_id",
                    "risk_level",
                    "affected_modules",
                    "recommended_reviewer",
                    "merge_decision"
                ]
            }
        ],
        "action_schema": {
            "episode_id": {"type": "string", "description": "From reset() response"},
            "risk_level": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            },
            "affected_modules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of affected module paths"
            },
            "recommended_reviewer": {
                "type": "string",
                "description": "Reviewer name from available_reviewers"
            },
            "merge_decision": {
                "type": "string",
                "enum": ["APPROVE", "BLOCK", "REQUEST_CHANGES"]
            }
        }
    }


@app.post("/grader")
async def grade_action(request: Dict[str, Any]):
    """Grade an arbitrary action against a known scenario.

    Request body:
        {
            "action": { ...CodeReviewAction fields... },
            "scenario_id": "task1_001"
        }

    Returns:
        { "score": 0.0-1.0, "feedback": "..." }
    """
    from server.dataset import DatasetLoader
    from server.graders import compute_reward, build_feedback

    action_data = request.get("action", {})
    scenario_id = request.get("scenario_id")

    if not action_data:
        raise HTTPException(status_code=400, detail="action required")
    if not scenario_id:
        raise HTTPException(status_code=400, detail="scenario_id required")

    # Load scenario by ID
    dataset = DatasetLoader()
    scenario = dataset.get_scenario_by_id(scenario_id)

    if not scenario:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario not found: {scenario_id}"
        )

    # Build action from request data
    action = CodeReviewAction(
        episode_id=action_data.get("episode_id", ""),
        risk_level=action_data.get("risk_level", "LOW"),
        affected_modules=action_data.get("affected_modules", []),
        recommended_reviewer=action_data.get("recommended_reviewer", ""),
        merge_decision=action_data.get("merge_decision", "")
    )

    # Grade and return
    score = compute_reward(action, scenario["ground_truth"], scenario["task"])
    feedback = build_feedback(action, scenario["ground_truth"], scenario["task"])

    return {"score": score, "feedback": feedback}


@app.get("/baseline")
async def get_baseline_scores():
    """Return pre-computed baseline scores from static file.

    NOTE: Scores are pre-computed and stored in baseline_scores.json.
    This endpoint never runs the agent inline — it just reads the file.
    This avoids blocking the server and state corruption.
    """
    baseline_path = Path(__file__).parent.parent / "baseline_scores.json"

    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            return json.load(f)

    # Fallback if file missing
    return {
        "baseline_scores": {
            "task1": 0.18,
            "task2": 0.12,
            "task3": 0.09
        },
        "agent": "deterministic_random",
        "seed": 42,
        "episodes_per_task": 100
    }


@app.get("/")
async def root():
    """Root endpoint with environment info."""
    return {
        "name": "codereview-env",
        "version": "1.0.0",
        "description": "RL environment for training code review intelligence agents",
        "tasks": ["task1", "task2", "task3"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks", "/grader", "/baseline"]
    }