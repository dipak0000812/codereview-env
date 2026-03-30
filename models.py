# models.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Import base classes from OpenEnv core
from core.env_server import Action, Observation, State


class CodeReviewAction(Action):
    """
    Sent by the agent to make a review decision.
    episode_id is mandatory — used to look up session state.
    """
    episode_id: str
    risk_level: str                  # LOW / MEDIUM / HIGH / CRITICAL
    affected_modules: List[str]      # Predicted blast radius
    recommended_reviewer: str        # Who should review this PR
    merge_decision: str              # APPROVE / BLOCK / REQUEST_CHANGES


class CodeReviewObservation(Observation):
    """
    Returned by reset() and step().
    Contains everything the agent needs to make a decision.
    """
    episode_id: str
    task: str                              # task1 / task2 / task3
    diff: str                              # The actual code diff
    dependency_map: Dict[str, List[str]]   # module → list of imports
    file_history: Dict[str, Any]           # commits, incidents, coverage
    available_reviewers: List[str]         # valid reviewer names
    done: bool                             # True after step()
    reward: float                          # 0.0 on reset, graded on step
    feedback: str                          # Grading breakdown message


class CodeReviewState(State):
    """
    Returned by state property.
    Minimal but spec-compliant.
    """
    episode_id: str
    step_count: int
    task: str