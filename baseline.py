"""Baseline agent for CodeReview environment.

A deterministic naive agent that serves as a baseline for comparison.
Always returns LOW risk, [] affected modules, alice, and APPROVE.
"""

import json
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    from client import CodeReviewAction
except ImportError:
    CodeReviewAction = None

class NaiveBaselineAgent:
    def predict(self, observation: Dict[str, Any], episode_id: str):
        revs = observation.get("available_reviewers", [])
        rev = revs[0] if revs else "alice"
        
        # If fallback without import, build a dummy object with to_dict
        if CodeReviewAction is not None:
            return CodeReviewAction(
                episode_id=episode_id,
                risk_level="LOW",
                affected_modules=[],
                recommended_reviewer=rev,
                merge_decision="APPROVE"
            )
        else:
            class DummyAction:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
                def to_dict(self):
                    return self.__dict__
            return DummyAction(
                episode_id=episode_id,
                risk_level="LOW",
                affected_modules=[],
                recommended_reviewer=rev,
                merge_decision="APPROVE"
            )

def run_baseline_inference(
    base_url: str = "http://localhost:7860",
    num_episodes: int = 10,
    tasks: List[str] = None,
    output_file: str = None
) -> Dict[str, Any]:
    if tasks is None:
        tasks = ["task1", "task2", "task3"]

    agent = NaiveBaselineAgent()
    results = {task: [] for task in tasks}

    try:
        import requests
        for task in tasks:
            print(f"\nEvaluating Task {task}...")
            task_scores = []

            for episode in range(num_episodes):
                response = requests.post(base_url + "/reset", json={"task": task}).json()
                obs = response["observation"]
                episode_id = obs["episode_id"]

                done = False
                rewards = []
                while not done:
                    action = agent.predict(obs, episode_id)
                    action_dict = action.to_dict()
                    r2 = requests.post(base_url + "/step", json={
                        "action": {
                            "episode_id": episode_id,
                            "risk_level": action_dict.get("risk_level", "LOW"),
                            "affected_modules": action_dict.get("affected_modules", []),
                            "recommended_reviewer": action_dict.get("recommended_reviewer", ""),
                            "merge_decision": action_dict.get("merge_decision", "")
                        }
                    }).json()
                    
                    obs = r2["observation"]
                    # For task3 with 3 steps, we collect the rewards across steps
                    # Reward usually accumulates over pieces, so total reward is sum
                    reward = r2.get("reward", 0.0)
                    rewards.append(reward)
                    done = r2.get("done", True)
                
                total_sc = sum(rewards)
                task_scores.append(total_sc)
                print(f"  Episode {episode + 1}: score = {total_sc:.3f}")

            avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            results[task] = task_scores
            results[f"{task}_avg"] = avg_score
            print(f"  Average: {avg_score:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to offline scoring...")

        from dataset import DatasetLoader
        from graders import compute_reward
        dataset = DatasetLoader()
        
        for task in tasks:
            print(f"\nEvaluating Task {task} (offline)...")
            task_scores = []

            for _ in range(num_episodes):
                scenario = dataset.sample(task)
                
                # In offline mode we must mimic the loop if we want an accurate total score, 
                # but compute_reward natively provides the compound score for the full scenario.
                # Since offline scoring is single-step composite in the dataset API:
                action = agent.predict(scenario, "offline_episode")
                
                score = compute_reward(
                    action,
                    scenario['ground_truth'],
                    task
                )
                task_scores.append(score)

            avg_score = sum(task_scores) / len(task_scores) if task_scores else 0.0
            results[task] = task_scores
            results[f"{task}_avg"] = avg_score
            print(f"  Average: {avg_score:.3f}")

    task_avgs = [results.get(f"{task}_avg", 0.0) for task in tasks]
    results["overall_avg"] = sum(task_avgs) / len(task_avgs) if task_avgs else 0.0

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results

def compute_baseline_scores(
    num_episodes: int = 100,
    tasks: List[str] = None
) -> Dict[str, float]:
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
