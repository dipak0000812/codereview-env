"""HTTP Client for CodeReview environment.

This client allows RL agents to interact with the CodeReview environment
via HTTP/WebSocket endpoints.
"""

import json
import httpx
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CodeReviewAction:
    """Action to send to the environment."""
    episode_id: str
    risk_level: str = "LOW"
    affected_modules: list = None
    recommended_reviewer: str = ""
    merge_decision: str = ""

    def __post_init__(self):
        if self.affected_modules is None:
            self.affected_modules = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "episode_id": self.episode_id,
            "risk_level": self.risk_level,
            "affected_modules": self.affected_modules,
            "recommended_reviewer": self.recommended_reviewer,
            "merge_decision": self.merge_decision
        }


@dataclass
class CodeReviewObservation:
    """Observation received from the environment."""
    episode_id: str
    task: str
    diff: str
    dependency_map: Dict[str, list]
    file_history: Dict[str, Any]
    available_reviewers: list
    done: bool
    reward: float
    feedback: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeReviewObservation':
        """Create observation from dictionary."""
        return cls(
            episode_id=data.get('episode_id', ''),
            task=data.get('task', ''),
            diff=data.get('diff', ''),
            dependency_map=data.get('dependency_map', {}),
            file_history=data.get('file_history', {}),
            available_reviewers=data.get('available_reviewers', []),
            done=data.get('done', False),
            reward=data.get('reward', 0.0),
            feedback=data.get('feedback', '')
        )


class HTTPEnvClient:
    """HTTP client for interacting with CodeReview environment.

    Usage:
        client = HTTPEnvClient("http://localhost:7860")

        # Reset environment
        obs, episode_id = client.reset("task1")

        # Take step
        action = CodeReviewAction(
            episode_id=episode_id,
            risk_level="HIGH",
            affected_modules=["auth/models.py"],
            recommended_reviewer="alice",
            merge_decision="BLOCK"
        )
        obs = client.step(action)

        # Get tasks info
        tasks = client.get_tasks()
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        """Initialize client.

        Args:
            base_url: Base URL of the environment server
        """
        self.base_url = base_url.rstrip('/')
        self.client = httpx.Client(timeout=30.0)

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def reset(self, task: str = "task1") -> Tuple[CodeReviewObservation, str]:
        """Reset the environment.

        Args:
            task: Task identifier (task1, task2, task3)

        Returns:
            Tuple of (observation, episode_id)
        """
        response = self.client.post(
            f"{self.base_url}/reset",
            json={"task": task}
        )
        response.raise_for_status()
        data = response.json()

        obs = CodeReviewObservation.from_dict(data['observation'])
        episode_id = data['episode_id']

        return obs, episode_id

    def step(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            Observation after taking the step
        """
        response = self.client.post(
            f"{self.base_url}/step",
            json={
                "action": action.to_dict(),
                "episode_id": action.episode_id
            }
        )
        response.raise_for_status()
        data = response.json()

        return CodeReviewObservation.from_dict(data['observation'])

    def get_tasks(self) -> Dict[str, Any]:
        """Get available tasks and action schema."""
        response = self.client.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    def get_baseline_scores(self) -> Dict[str, float]:
        """Get pre-computed baseline scores."""
        response = self.client.get(f"{self.base_url}/baseline")
        response.raise_for_status()
        return response.json()

    def grade(self, action: Dict, scenario_id: str) -> Dict[str, Any]:
        """Grade an action against a specific scenario.

        Args:
            action: Action dictionary
            scenario_id: Scenario ID to grade against

        Returns:
            Dictionary with 'score' and 'feedback'
        """
        response = self.client.post(
            f"{self.base_url}/grader",
            json={
                "action": action,
                "scenario_id": scenario_id
            }
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
