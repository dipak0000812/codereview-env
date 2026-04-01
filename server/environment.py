"""Stateless CodeReview Environment for OpenEnv.

CRITICAL: This environment is STATELESS. All episode state lives in
sessions.py keyed by episode_id. The environment object never holds
episode state — it only reads/writes to the session store.

This ensures concurrent requests never contaminate each other's episodes.

SUPPORTS_CONCURRENT_SESSIONS = True because we use session store isolation.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .sessions import create_session, get_session, close_session
from .dataset import DatasetLoader
from .graders import compute_reward, build_feedback

try:
    from models import CodeReviewObservation, CodeReviewState, CodeReviewAction
except ImportError:
    from ..models import CodeReviewObservation, CodeReviewState, CodeReviewAction


# Global dataset instance — loaded once, shared (read-only, no race conditions)
dataset = DatasetLoader()


class CodeReviewEnvironment(Environment):
    """Stateless RL environment for code review intelligence.

    Trains RL agents to review code like senior engineers:
    - Assess risk level (LOW/MEDIUM/HIGH/CRITICAL)
    - Compute blast radius from dependency graphs
    - Identify optimal reviewers
    - Make merge decisions (APPROVE/BLOCK/REQUEST_CHANGES)
    """

    # Enable concurrent sessions — we use session store isolation
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize environment. No episode state stored here."""
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task: str = "task1", **kwargs) -> CodeReviewObservation:
        """Reset the environment and start a new episode.

        NOTE: OpenEnv spec requires reset() to return just the observation.
        The episode_id is embedded inside the observation for our session store.

        Args:
            task: Task identifier (task1, task2, task3)

        Returns:
            CodeReviewObservation with episode_id embedded
        """
        # Sample a scenario for this task
        scenario = dataset.sample(task)

        # Create new session — returns UUID episode_id
        episode_id = create_session(task, scenario)

        # Update internal state
        self._state = State(episode_id=episode_id, step_count=0)

        return CodeReviewObservation(
            episode_id=episode_id,
            task=task,
            diff=scenario["diff"],
            dependency_map=scenario["dependency_map"],
            file_history=scenario.get("file_history", {}),
            available_reviewers=scenario["available_reviewers"],
            done=False,
            reward=0.0,
            feedback="Analyze the diff and make your review decision."
        )

    def step(self, action: CodeReviewAction, **kwargs) -> CodeReviewObservation:
        """Take a step in the environment.

        The episode_id is read from action.episode_id to look up session state.

        Args:
            action: Agent's CodeReviewAction (contains episode_id)

        Returns:
            CodeReviewObservation with reward and done=True
        """
        episode_id = action.episode_id

        # Retrieve session
        session = get_session(episode_id)
        scenario = session["scenario"]
        task = session["task"]

        # Compute reward
        reward = compute_reward(action, scenario["ground_truth"], task)

        # Build human-readable feedback
        feedback = build_feedback(action, scenario["ground_truth"], task)

        # Close session — single-step episode is done
        close_session(episode_id)

        # Update internal state
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
            feedback=feedback
        )

    @property
    def state(self) -> State:
        """Return current environment state."""
        return self._state