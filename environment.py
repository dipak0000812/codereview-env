"""Stateless CodeReview Environment for OpenEnv.

Implements the Environment interface from openenv-core.
All state lives in sessions.py (UUID-keyed dict), never in this class,
ensuring SUPPORTS_CONCURRENT_SESSIONS = True is safe.
"""

import sys
from pathlib import Path
from typing import Optional, Any
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from sessions import create_session, get_session, close_session
from dataset import DatasetLoader
from graders import (
    _clamp, compute_reward, build_feedback,
    risk_score, jaccard_score, reviewer_score, merge_score
)
from models import CodeReviewObservation, CodeReviewState, CodeReviewAction

dataset = DatasetLoader()


class CodeReviewEnvironment(Environment):
    """Code review RL environment.

    Stateless — all episode data is stored in sessions.py keyed by UUID.
    Supports concurrent sessions safely.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = CodeReviewState(episode_id="", step_count=0, task="")

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, task: str = "task1", episode_id: Optional[str] = None, **kwargs) -> CodeReviewObservation:
        """Reset environment and return initial observation.

        Args:
            task: One of task1, task2, task3
            episode_id: Optional UUID to sync with server-generated ID
            **kwargs: Forwarded but not used

        Returns:
            CodeReviewObservation (single object per OpenEnv spec)
        """
        scenario = dataset.sample(task)

        # Sync with server-generated episode_id if provided
        episode_id = create_session(task, scenario, episode_id=episode_id)

        session = get_session(episode_id)
        max_steps = 3 if task == "task3" else 1
        session["max_steps"] = max_steps
        session["step_count"] = 0

        self._state = CodeReviewState(episode_id=episode_id, step_count=0, task=task)

        # For task3: reveal info progressively (diff only at step 0)
        return CodeReviewObservation(
            episode_id=episode_id,
            task=task,
            diff=scenario["diff"],
            dependency_map=scenario["dependency_map"] if task != "task3" else {},
            file_history=scenario.get("file_history", {}) if task != "task3" else {},
            available_reviewers=scenario["available_reviewers"] if task != "task3" else [],
            done=False,
            reward=0.0,
            feedback=(
                f"Analyze the diff and assign a risk level. (Step 1/{max_steps})"
                if task == "task3" else
                "Analyze the diff and complete the review action."
            )
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: CodeReviewAction, **kwargs) -> CodeReviewObservation:
        """Advance the episode by one step.

        Args:
            action: CodeReviewAction with episode_id and review fields
            **kwargs: Forwarded but not used

        Returns:
            CodeReviewObservation (done=True signals episode end)
        """
        episode_id = action.episode_id
        session = get_session(episode_id)  # raises ValueError if not found
        step = session["step_count"]
        task = session["task"]
        scenario = session["scenario"]
        gt = scenario["ground_truth"]
        max_steps = session["max_steps"]

        if task == "task3":
            return self._step_task3(action, session, step, scenario, gt, episode_id)
        else:
            return self._step_single(action, session, task, scenario, gt, episode_id)

    def _step_task3(self, action, session, step, scenario, gt, episode_id):
        """Multi-step logic for task3."""
        task = "task3"
        if step == 0:
            # Step 1: Score risk classification (25% weight)
            raw = risk_score(action.risk_level, gt["risk_level"]) * 0.25
            reward = _clamp(raw)
            session["step_count"] = 1
            self._state = CodeReviewState(episode_id=episode_id, step_count=1, task=task)
            return CodeReviewObservation(
                episode_id=episode_id, task=task,
                diff=scenario["diff"],
                dependency_map=scenario["dependency_map"],
                file_history={},
                available_reviewers=[],
                done=False, reward=reward,
                feedback=(
                    f"Step 1/3 complete — risk_score={reward:.3f} "
                    f"(predicted={action.risk_level}, truth={gt['risk_level']}). "
                    "Next: identify affected modules (blast radius)."
                )
            )

        elif step == 1:
            # Step 2: Score blast radius (30% weight)
            raw = jaccard_score(action.affected_modules, gt.get("blast_radius", [])) * 0.30
            reward = _clamp(raw)
            session["step_count"] = 2
            self._state = CodeReviewState(episode_id=episode_id, step_count=2, task=task)
            return CodeReviewObservation(
                episode_id=episode_id, task=task,
                diff=scenario["diff"],
                dependency_map=scenario["dependency_map"],
                file_history=scenario.get("file_history", {}),
                available_reviewers=scenario["available_reviewers"],
                done=False, reward=reward,
                feedback=(
                    f"Step 2/3 complete — blast_radius_score={reward:.3f}. "
                    "Next: assign reviewer and merge decision."
                )
            )

        else:
            # Step 3: Score reviewer (20%) + merge decision (25%)
            rev_sc = reviewer_score(action.recommended_reviewer, gt["recommended_reviewer"])
            mrg_sc = merge_score(action.merge_decision, gt["merge_decision"], gt["risk_level"])
            raw = (rev_sc * 0.20) + (mrg_sc * 0.25)
            reward = _clamp(raw)
            session["step_count"] = 3
            self._state = CodeReviewState(episode_id=episode_id, step_count=3, task=task)
            # Clean up session after final step
            close_session(episode_id)
            return CodeReviewObservation(
                episode_id=episode_id, task=task,
                diff=scenario["diff"],
                dependency_map=scenario["dependency_map"],
                file_history=scenario.get("file_history", {}),
                available_reviewers=scenario["available_reviewers"],
                done=True, reward=reward,
                feedback=(
                    f"Step 3/3 complete — reviewer_score={rev_sc:.3f}, "
                    f"merge_score={mrg_sc:.3f}, final_reward={reward:.3f}. "
                    "Episode finished."
                )
            )

    def _step_single(self, action, session, task, scenario, gt, episode_id):
        """Single-step logic for task1 and task2."""
        reward = compute_reward(action, gt, task)
        reward = _clamp(float(reward))
        feedback = build_feedback(action, gt, task)
        session["step_count"] = 1
        self._state = CodeReviewState(episode_id=episode_id, step_count=1, task=task)
        close_session(episode_id)
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

    # ------------------------------------------------------------------
    # grader
    # ------------------------------------------------------------------
    def grader(self, task: str, episode_id: Optional[str] = None, actions: list = None, **kwargs) -> float:
        """Standard OpenEnv grader. Returns float strictly in (0, 1).

        The validator calls this after the episode. We look up the session
        scenario if available, else fall back to sampling a new one.

        Args:
            task: Task identifier
            episode_id: Episode UUID (may be expired from session store)
            actions: List of action dicts submitted during the episode
            **kwargs: Forwarded but not used

        Returns:
            float strictly in open interval (0, 1) — never 0.0 or 1.0
        """
        if not actions:
            return _clamp(0.0)

        # Try to use session scenario; fall back to a fresh sample
        try:
            session = get_session(episode_id)
            scenario = session["scenario"]
            gt = scenario["ground_truth"]
        except (ValueError, TypeError, KeyError):
            scenario = dataset.sample(task)
            gt = scenario["ground_truth"]

        # Use the last action submitted in the episode
        action_dict = actions[-1]
        if not isinstance(action_dict, dict):
            try:
                action_dict = action_dict.model_dump()
            except AttributeError:
                try:
                    action_dict = vars(action_dict)
                except Exception:
                    action_dict = {}

        action = CodeReviewAction(
            episode_id=episode_id or str(uuid4()),
            risk_level=action_dict.get("risk_level", "LOW"),
            affected_modules=action_dict.get("affected_modules", []),
            recommended_reviewer=action_dict.get("recommended_reviewer", ""),
            merge_decision=action_dict.get("merge_decision", ""),
        )

        score = compute_reward(action, gt, task)
        score = _clamp(float(score))
        # Safety: never let a raw assertion crash the grader
        score = max(1e-6, min(1.0 - 1e-6, score))
        return score

    # ------------------------------------------------------------------
    # state / metadata / close
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> dict:
        """Return environment metadata for introspection."""
        return {
            "name": "codereview-env",
            "version": "2.0.0",
            "description": "RL environment for training agents to perform intelligent code review.",
            "tasks": ["task1", "task2", "task3"],
            "supports_concurrent_sessions": self.SUPPORTS_CONCURRENT_SESSIONS,
        }

    def close(self) -> None:
        """Cleanup resources (nothing to tear down for stateless env)."""
        self._state = CodeReviewState(episode_id="", step_count=0, task="")