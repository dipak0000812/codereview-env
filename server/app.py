"""FastAPI server for CodeReview environment."""

import json
from pathlib import Path
from typing import Dict, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

import sys
import os

# Add working directory to path so root modules are importable
sys.path.insert(0, os.getcwd())

from models import CodeReviewAction, CodeReviewObservation
from environment import CodeReviewEnvironment
from dataset import DatasetLoader
from graders import compute_reward, build_feedback
from sessions import purge_stale_sessions, session_count

app = create_app(
    env=CodeReviewEnvironment,
    action_cls=CodeReviewAction,
    observation_cls=CodeReviewObservation,
    env_name="codereview-env",
    max_concurrent_envs=100,
)


@app.get("/")
async def root():
    return {
        "name": "codereview-env",
        "version": "2.0.0",
        "description": "RL environment for intelligent multi-step code review",
        "tasks": ["task1", "task2", "task3"],
        "endpoints": [
            "/reset", "/step", "/state", "/health",
            "/tasks", "/grader", "/baseline", "/info", "/metrics"
        ],
    }


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": [
            {
                "name": "task1",
                "description": "Classify risk level from code diff",
                "difficulty": "easy",
                "steps": 1,
                "action_fields": ["episode_id", "risk_level"],
            },
            {
                "name": "task2",
                "description": "Identify blast radius from dependency map",
                "difficulty": "medium",
                "steps": 1,
                "action_fields": ["episode_id", "affected_modules"],
            },
            {
                "name": "task3",
                "description": "Full multi-step review: risk → blast radius → reviewer + merge decision",
                "difficulty": "hard",
                "steps": 3,
                "action_fields": [
                    "episode_id", "risk_level",
                    "affected_modules", "recommended_reviewer", "merge_decision",
                ],
            },
        ],
        "action_schema": {
            "episode_id": {"type": "string", "required": True},
            "risk_level": {
                "type": "string",
                "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            },
            "affected_modules": {"type": "array", "items": {"type": "string"}},
            "recommended_reviewer": {"type": "string"},
            "merge_decision": {
                "type": "string",
                "enum": ["APPROVE", "BLOCK", "REQUEST_CHANGES"],
            },
        },
    }


@app.get("/baseline")
async def get_baseline_scores():
    baseline_path = Path(__file__).parent.parent / "baseline_scores.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)
    return {
        "baseline_scores": {
            "task1": 0.394,
            "task2": 0.001,
            "task3": 0.283,
        },
        "agent": "NaiveBaselineAgent (always predicts LOW, [], alice, APPROVE)",
        "note": "Scores above baseline indicate agent improvement.",
    }


@app.get("/info")
async def get_info():
    """Return environment metadata including dataset statistics."""
    loader = DatasetLoader()
    return {
        "name": "codereview-env",
        "version": "2.0.0",
        "dataset": {
            "task1_count": loader.get_task_count("task1"),
            "task2_count": loader.get_task_count("task2"),
            "task3_count": loader.get_task_count("task3"),
            "total": loader.get_total_count(),
        },
        "reward_design": {
            "task1": "risk_score (ordinal distance with safety penalty)",
            "task2": "jaccard_similarity (case-insensitive set overlap)",
            "task3": "composite: 0.25*risk + 0.30*blast_radius + 0.20*reviewer + 0.25*merge",
        },
        "grader_bounds": "strict open interval (0, 1) — never 0.0 or 1.0 exactly",
        "concurrent_sessions": True,
    }


@app.get("/metrics")
async def get_metrics():
    """Return server health metrics."""
    purged = purge_stale_sessions()
    return {
        "active_sessions": session_count(),
        "sessions_purged": purged,
        "status": "healthy",
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()