"""Baseline agent for CodeReview environment.

A deterministic random agent that serves as a baseline for comparison.
Uses seed=42 for reproducibility.

This agent:
- Makes random but deterministic predictions
- Does NOT require any API keys
- Can run offline without network access
"""

import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

# Try to import client, fall back to direct scoring if unavailable
try:
    from client import HTTPEnvClient, CodeReviewAction
except ImportError:
    from server.graders import compute_reward
    from server.dataset import DatasetLoader

    CodeReviewAction = None


@dataclass
class BaselineAction:
    """Action produced by baseline agent."""
    episode_id: str
    risk_level: str
    affected_modules: List[str]
    recommended_reviewer: str
    merge_decision: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "risk_level": self.risk_level,
            "affected_modules": self.affected_modules,
            "recommended_reviewer": self.recommended_reviewer,
            "merge_decision": self.merge_decision
        }


class BaselineAgent:
    """Deterministic random baseline agent.

    Makes random but reproducible predictions based on task type.
    Uses seed=42 for deterministic behavior.
    """

    RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    MERGE_DECISIONS = ["APPROVE", "BLOCK", "REQUEST_CHANGES"]
    DEFAULT_REVIEWERS = ["alice", "bob", "charlie", "david"]

    def __init__(self, seed: int = 42):
        """Initialize baseline agent.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def predict(self, observation: Dict[str, Any], episode_id: str) -> BaselineAction:
        """Make a prediction based on observation.

        Args:
            observation: Environment observation
            episode_id: Episode ID

        Returns:
            BaselineAction with predictions
        """
        task = observation.get('task', 'task1')
        available_reviewers = observation.get('available_reviewers', self.DEFAULT_REVIEWERS)
        dependency_map = observation.get('dependency_map', {})

        if task == 'task1':
            # Task 1: Risk Classification (Easy)
            return self._predict_task1(episode_id, observation)

        elif task == 'task2':
            # Task 2: Blast Radius Identification (Medium)
            return self._predict_task2(episode_id, dependency_map)

        elif task == 'task3':
            # Task 3: Full Review Decision (Hard)
            return self._predict_task3(episode_id, observation, available_reviewers, dependency_map)

        # Default fallback
        return BaselineAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=[],
            recommended_reviewer=available_reviewers[0] if available_reviewers else "alice",
            merge_decision="APPROVE"
        )

    def _predict_task1(self, episode_id: str, observation: Dict) -> BaselineAction:
        """Predict for Task 1 (Risk Classification)."""
        # Random risk level
        risk_level = random.choice(self.RISK_LEVELS)

        return BaselineAction(
            episode_id=episode_id,
            risk_level=risk_level,
            affected_modules=[],
            recommended_reviewer="",
            merge_decision=""
        )

    def _predict_task2(self, episode_id: str, dependency_map: Dict) -> BaselineAction:
        """Predict for Task 2 (Blast Radius)."""
        # Random subset of modules from dependency map
        all_modules = list(dependency_map.keys())
        for deps in dependency_map.values():
            all_modules.extend(deps)

        all_modules = list(set(all_modules))

        if not all_modules:
            affected_modules = []
        else:
            # Random subset (1 to all)
            num_modules = random.randint(1, min(len(all_modules), 5))
            affected_modules = random.sample(all_modules, num_modules)

        return BaselineAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=affected_modules,
            recommended_reviewer="",
            merge_decision=""
        )

    def _predict_task3(self, episode_id: str, observation: Dict,
                       available_reviewers: List[str], dependency_map: Dict) -> BaselineAction:
        """Predict for Task 3 (Full Review)."""
        # Random risk level
        risk_level = random.choice(self.RISK_LEVELS)

        # Random subset of modules
        all_modules = list(dependency_map.keys())
        for deps in dependency_map.values():
            all_modules.extend(deps)
        all_modules = list(set(all_modules))

        if not all_modules:
            affected_modules = []
        else:
            num_modules = random.randint(1, min(len(all_modules), 5))
            affected_modules = random.sample(all_modules, num_modules)

        # Random reviewer
        recommended_reviewer = random.choice(available_reviewers) if available_reviewers else "alice"

        # Random merge decision
        merge_decision = random.choice(self.MERGE_DECISIONS)

        return BaselineAction(
            episode_id=episode_id,
            risk_level=risk_level,
            affected_modules=affected_modules,
            recommended_reviewer=recommended_reviewer,
            merge_decision=merge_decision
        )


def run_baseline_inference(
    base_url: str = "http://localhost:7860",
    num_episodes: int = 10,
    tasks: List[str] = None,
    output_file: str = None
) -> Dict[str, Any]:
    """Run baseline agent against the environment.

    Args:
        base_url: Environment server URL
        num_episodes: Number of episodes to run
        tasks: List of tasks to evaluate (defaults to all)
        output_file: Optional file to save results

    Returns:
        Dictionary with baseline scores
    """
    if tasks is None:
        tasks = ["task1", "task2", "task3"]

    agent = BaselineAgent(seed=42)
    results = {task: [] for task in tasks}

    try:
        with HTTPEnvClient(base_url) as client:
            for task in tasks:
                print(f"\nEvaluating Task {task}...")
                task_scores = []

                for episode in range(num_episodes):
                    # Reset
                    obs, episode_id = client.reset(task)

                    # Predict
                    action = agent.predict(obs.__dict__, episode_id)

                    # Step
                    new_obs = client.step(CodeReviewAction(
                        episode_id=episode_id,
                        risk_level=action.risk_level,
                        affected_modules=action.affected_modules,
                        recommended_reviewer=action.recommended_reviewer,
                        merge_decision=action.merge_decision
                    ))

                    score = new_obs.reward
                    task_scores.append(score)
                    print(f"  Episode {episode + 1}: score = {score:.3f}")

                avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
                results[task] = task_scores
                results[f"{task}_avg"] = avg_score
                print(f"  Average: {avg_score:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to offline scoring...")

        # Offline fallback using dataset loader
        from server.dataset import DatasetLoader
        from server.graders import compute_reward
        dataset = DatasetLoader()
        agent = BaselineAgent(seed=42)

        for task in tasks:
            print(f"\nEvaluating Task {task} (offline)...")
            task_scores = []

            for _ in range(num_episodes):
                scenario = dataset.sample(task)
                action = agent.predict(scenario, "offline_episode")
                action.episode_id = "offline_episode"

                score = compute_reward(
                    type('Action', (), action.to_dict())(),
                    scenario['ground_truth'],
                    task
                )
                task_scores.append(score)

            avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            results[task] = task_scores
            results[f"{task}_avg"] = avg_score
            print(f"  Average: {avg_score:.3f}")

    # Calculate overall score
    task_avgs = [results.get(f"{task}_avg", 0.0) for task in tasks]
    results["overall_avg"] = sum(task_avgs) / len(task_avgs) if task_avgs else 0.0

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


def compute_baseline_scores(
    num_episodes: int = 100,
    tasks: List[str] = None
) -> Dict[str, float]:
    """Compute baseline scores for all tasks.

    Args:
        num_episodes: Number of episodes for averaging
        tasks: Tasks to evaluate

    Returns:
        Dictionary mapping task -> average score
    """
    if tasks is None:
        tasks = ["task1", "task2", "task3"]

    results = run_baseline_inference(
        base_url="http://localhost:7860",
        num_episodes=num_episodes,
        tasks=tasks
    )

    baseline_scores = {}
    for task in tasks:
        baseline_scores[task] = results.get(f"{task}_avg", 0.0)

    return baseline_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline agent")
    parser.add_argument("--url", default="http://localhost:7860", help="Environment URL")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"], help="Tasks to evaluate")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    results = run_baseline_inference(
        base_url=args.url,
        num_episodes=args.episodes,
        tasks=args.tasks,
        output_file=args.output
    )

    print(f"\n{'='*50}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'='*50}")
    for task in args.tasks:
        avg = results.get(f"{task}_avg", 0.0)
        print(f"  {task}: {avg:.3f}")
    print(f"  Overall: {results.get('overall_avg', 0.0):.3f}")
