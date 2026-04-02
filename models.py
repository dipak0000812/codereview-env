"""Pydantic models for CodeReview environment."""

from typing import Dict, List, Any
from pydantic import Field
from openenv.core import Action, Observation, State


class CodeReviewAction(Action):
    episode_id: str = Field(..., description="Mandatory — session lookup key")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    affected_modules: List[str] = Field(default_factory=list)
    recommended_reviewer: str = Field(default="")
    merge_decision: str = Field(default="")


class CodeReviewObservation(Observation):
    episode_id: str = Field(default="")
    task: str = Field(default="")
    diff: str = Field(default="")
    dependency_map: Dict[str, List[str]] = Field(default_factory=dict)
    file_history: Dict[str, Any] = Field(default_factory=dict)
    available_reviewers: List[str] = Field(default_factory=list)
    feedback: str = Field(default="")


class CodeReviewState(State):
    task: str = Field(default="")