"""Pydantic models for CodeReview environment.

Defines Action, Observation, and State types for RL agents
to interact with the code review environment.

CRITICAL: Inherits from OpenEnv base classes for spec compliance.
Correct imports: openenv.core.env_server.types
"""

from typing import Dict, List, Any, Optional
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class CodeReviewAction(Action):
    """Action sent by the agent to make a review decision.

    episode_id is mandatory - used to look up session state.
    Inherits 'metadata' field from Action base class.
    """
    episode_id: str = Field(..., description="Mandatory — session lookup key")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    affected_modules: List[str] = Field(
        default_factory=list,
        description="Predicted blast radius modules"
    )
    recommended_reviewer: str = Field(
        default="",
        description="Who should review this PR"
    )
    merge_decision: str = Field(
        default="",
        description="APPROVE / BLOCK / REQUEST_CHANGES"
    )


class CodeReviewObservation(Observation):
    """Observation returned by reset() and step().

    Contains everything the agent needs to make a decision.
    Inherits 'done', 'reward', 'metadata' fields from Observation base class.
    """
    episode_id: str = Field(default="", description="Episode identifier")
    task: str = Field(default="", description="task1 / task2 / task3")
    diff: str = Field(default="", description="The actual code diff")
    dependency_map: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Module → imports list"
    )
    file_history: Dict[str, Any] = Field(
        default_factory=dict,
        description="Commits, incidents, coverage per file"
    )
    available_reviewers: List[str] = Field(
        default_factory=list,
        description="Valid reviewer names"
    )
    feedback: str = Field(
        default="",
        description="Human-readable grading breakdown"
    )


class CodeReviewState(State):
    """State returned by state property.

    Inherits 'episode_id' and 'step_count' from State base class.
    """
    task: str = Field(default="", description="Current task identifier")


# ── Helper models for custom endpoints ────────────────────────────────────────

class TaskInfo:
    """Task metadata for /tasks endpoint."""
    def __init__(self, name, description, difficulty, action_fields):
        self.name = name
        self.description = description
        self.difficulty = difficulty
        self.action_fields = action_fields

    def dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "action_fields": self.action_fields
        }