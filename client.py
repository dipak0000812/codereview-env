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
    """Asynchronous HTTP client for interacting with CodeReview environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        """Initialize client."""
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def reset(self, task: str = "task1") -> Tuple[CodeReviewObservation, str]:
        """Reset the environment."""
        response = await self.client.post(
            f"{self.base_url}/reset",
            json={"task": task}
        )
        response.raise_for_status()
        data = response.json()
        
        obs_dict = data['observation']
        # Merge reward/done from top level
        obs_dict['reward'] = data.get('reward', 0.0)
        obs_dict['done'] = data.get('done', False)
        
        obs = CodeReviewObservation.from_dict(obs_dict)
        episode_id = obs_dict.get('episode_id', '')

        return obs, episode_id

    async def step(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Take a step in the environment."""
        response = await self.client.post(
            f"{self.base_url}/step",
            json={
                "action": action.to_dict(),
                "episode_id": action.episode_id
            }
        )
        response.raise_for_status()
        data = response.json()

        obs_dict = data['observation']
        # Merge reward/done from top level
        obs_dict['reward'] = data.get('reward', 0.0)
        obs_dict['done'] = data.get('done', False)

        return CodeReviewObservation.from_dict(obs_dict)

    async def get_tasks(self) -> Dict[str, Any]:
        """Get available tasks and action schema."""
        response = await self.client.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()

    async def get_baseline_scores(self) -> Dict[str, float]:
        """Get pre-computed baseline scores."""
        response = await self.client.get(f"{self.base_url}/baseline")
        response.raise_for_status()
        return response.json()

    async def grade(self, action: Dict, scenario_id: str) -> Dict[str, Any]:
        """Grade an action against a specific scenario."""
        response = await self.client.post(
            f"{self.base_url}/grader",
            json={
                "action": action,
                "scenario_id": scenario_id
            }
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
