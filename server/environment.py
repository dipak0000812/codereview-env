"""Stateless CodeReview Environment for OpenEnv."""

import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core import Environment, State

from .sessions import create_session, get_session, close_session
from .dataset import DatasetLoader
from .graders import compute_reward, build_feedback
from models import CodeReviewObservation, CodeReviewState, CodeReviewAction

dataset = DatasetLoader()


class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State(episode_id="", step_count=0)

    def reset(self, task: str = "task1", episode_id: Optional[str] = None, **kwargs) -> CodeReviewObservation:
        """Reset environment and return observation."""
        scenario = dataset.sample(task)
        
        # Synchronize with server-generated episode_id if provided
        episode_id = create_session(task, scenario, episode_id=episode_id)
        
        self._state = State(episode_id=episode_id, step_count=0)

        obs = CodeReviewObservation(
            episode_id=episode_id,
            task=task,
            diff=scenario["diff"],
            dependency_map=scenario["dependency_map"],
            file_history=scenario.get("file_history", {}),
            available_reviewers=scenario["available_reviewers"],
            done=False,
            reward=0.0,
            feedback="Analyze the diff and make your review decision.",
        )
        return obs

    def step(self, action: CodeReviewAction, episode_id: str, **kwargs) -> CodeReviewObservation:
        session = get_session(episode_id)
        scenario = session["scenario"]
        task = session["task"]

        reward = compute_reward(action, scenario["ground_truth"], task)
        feedback = build_feedback(action, scenario["ground_truth"], task)
        close_session(episode_id)

        self._state = State(episode_id=episode_id, step_count=1)

        return CodeReviewObservation(
            episode_id=episode_id,
            task=task,
            diff=scenario["diff"],
            dependency_map=scenario["dependency_map"],
            file_history=scenario.get("file_history", {}),
            available_reviewers=scenario["available_reviewers"],
            done=True,
            reward=reward,
            feedback=feedback,
        )

    @property
    def state(self) -> State:
        return self._state