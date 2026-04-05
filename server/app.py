"""FastAPI server for CodeReview environment."""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import HTTPException
from openenv.core.env_server.http_server import create_app

import sys
import os
# Add working directory to path so root modules (models, environment, etc.) are importable
# This is needed because the package is installed to site-packages but root modules live in /app
sys.path.insert(0, os.getcwd())

from models import CodeReviewAction, CodeReviewObservation
from environment import CodeReviewEnvironment
from dataset import DatasetLoader
from graders import compute_reward, build_feedback

app = create_app(
    env=CodeReviewEnvironment,
    action_cls=CodeReviewAction,
    observation_cls=CodeReviewObservation,
    env_name="codereview-env",
    max_concurrent_envs=100,
)


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": [
            {
                "name": "task1",
                "description": "Classify risk level from code diff",
                "difficulty": "easy",
                "action_fields": ["episode_id", "risk_level"],
            },
            {
                "name": "task2",
                "description": "Identify blast radius from dependency map",
                "difficulty": "medium",
                "action_fields": ["episode_id", "affected_modules"],
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
                    "merge_decision",
                ],
            },
        ],
        "action_schema": {
            "episode_id": {"type": "string"},
            "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
            "affected_modules": {"type": "array", "items": {"type": "string"}},
            "recommended_reviewer": {"type": "string"},
            "merge_decision": {"type": "string", "enum": ["APPROVE", "BLOCK", "REQUEST_CHANGES"]},
        },
    }


@app.post("/grader")
async def grade_action(request: Dict[str, Any]):
    action_data = request.get("action", {})
    scenario_id = request.get("scenario_id")
    if not action_data or not scenario_id:
        raise HTTPException(status_code=400, detail="action and scenario_id required")

    dataset = DatasetLoader()
    scenario = dataset.get_scenario_by_id(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {scenario_id}")

    action = CodeReviewAction(
        episode_id=action_data.get("episode_id", ""),
        risk_level=action_data.get("risk_level", "LOW"),
        affected_modules=action_data.get("affected_modules", []),
        recommended_reviewer=action_data.get("recommended_reviewer", ""),
        merge_decision=action_data.get("merge_decision", ""),
    )

    score = compute_reward(action, scenario["ground_truth"], scenario["task"])
    feedback = build_feedback(action, scenario["ground_truth"], scenario["task"])

    return {"score": score, "feedback": feedback}


@app.get("/baseline")
async def get_baseline_scores():
    baseline_path = Path(__file__).parent.parent / "baseline_scores.json"
    if baseline_path.exists():
        with open(baseline_path, "r") as f:
            return json.load(f)
    return {"baseline_scores": {"task1": 0.18, "task2": 0.12, "task3": 0.09}, "agent": "deterministic_random", "seed": 42}


@app.get("/")
async def root():
    return {
        "name": "codereview-env",
        "version": "1.0.0",
        "tasks": ["task1", "task2", "task3"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks", "/grader", "/baseline"],
    }


def main():
    import uvicorn
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()