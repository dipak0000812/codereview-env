"""Stateless CodeReview Environment for OpenEnv."""

import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from sessions import create_session, get_session, close_session
from dataset import DatasetLoader
from graders import compute_reward, build_feedback, risk_score, jaccard_score, reviewer_score, merge_score
from models import CodeReviewObservation, CodeReviewState, CodeReviewAction

dataset = DatasetLoader()


class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = CodeReviewState(episode_id="", step_count=0, task="")

    def reset(self, task: str = "task1", episode_id: Optional[str] = None, **kwargs) -> tuple:
        """Reset environment and return observation and episode_id."""
        scenario = dataset.sample(task)
        
        # Synchronize with server-generated episode_id if provided
        episode_id = create_session(task, scenario, episode_id=episode_id)
        
        session = get_session(episode_id)
        max_steps = 3 if task == "task3" else 1
        session["max_steps"] = max_steps
        session["step_count"] = 0

        self._state = CodeReviewState(episode_id=episode_id, step_count=0, task=task)

        obs = CodeReviewObservation(
            episode_id=episode_id,
            task=task,
            diff=scenario["diff"],
            dependency_map=scenario["dependency_map"],
            file_history=scenario.get("file_history", {}),
            available_reviewers=scenario["available_reviewers"],
            done=False,
            reward=0.0,
            feedback=f"Analyze the diff. Step 1/{max_steps}" if task == "task3" else "Analyze the diff."
        )
        return obs, episode_id

    def step(self, action: CodeReviewAction, **kwargs) -> CodeReviewObservation:
        episode_id = action.episode_id
        session = get_session(episode_id)
        step = session["step_count"]
        task = session["task"]
        scenario = session["scenario"]
        gt = scenario["ground_truth"]
        max_steps = session["max_steps"]

        if task == "task3":
            if step == 0:
                reward = risk_score(action.risk_level, gt["risk_level"]) * 0.25
                next_obs = CodeReviewObservation(
                    episode_id=episode_id, task=task,
                    diff=scenario["diff"],
                    dependency_map=scenario["dependency_map"],
                    file_history={},
                    available_reviewers=[],
                    done=False, reward=reward,
                    feedback=f"Step 1/3: risk_score={reward:.2f}. Next: Blast radius."
                )
            elif step == 1:
                reward = jaccard_score(action.affected_modules, gt.get("blast_radius", [])) * 0.30
                next_obs = CodeReviewObservation(
                    episode_id=episode_id, task=task,
                    diff=scenario["diff"],
                    dependency_map=scenario["dependency_map"],
                    file_history=scenario.get("file_history", {}),
                    available_reviewers=scenario["available_reviewers"],
                    done=False, reward=reward,
                    feedback=f"Step 2/3: blast_radius_score={reward:.2f}. Next: Final decision."
                )
            else:
                rev_score = reviewer_score(action.recommended_reviewer, gt["recommended_reviewer"])
                merge_sc = merge_score(action.merge_decision, gt["merge_decision"], gt["risk_level"])
                # Step 3: Final decision (reviewer 20% + merge 25%)
                reward = (rev_score * 0.20) + (merge_sc * 0.25)
                next_obs = CodeReviewObservation(
                    episode_id=episode_id, task=task,
                    diff=scenario["diff"],
                    dependency_map=scenario["dependency_map"],
                    file_history=scenario.get("file_history", {}),
                    available_reviewers=scenario["available_reviewers"],
                    done=True, reward=reward,
                    feedback=f"Step 3/3: final_decision_score={reward:.2f}. Episode finished."
                )
                close_session(episode_id)
            session["step_count"] += 1
            self._state = CodeReviewState(episode_id=episode_id, step_count=session["step_count"], task=task)
            return next_obs
        else:
            reward = compute_reward(action, gt, task)
            feedback = build_feedback(action, gt, task)
            close_session(episode_id)
            self._state = CodeReviewState(episode_id=episode_id, step_count=1, task=task)
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