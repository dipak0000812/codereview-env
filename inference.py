#!/usr/bin/env python3
"""
Inference script for CodeReview environment.
Conforms to required stdout format for hackathon evaluation.
"""

import asyncio
import os
import json
import sys
from typing import Optional

from openai import OpenAI

# Import your environment client and models
from client import HTTPEnvClient, CodeReviewAction
from models import CodeReviewObservation

# ----------------------------------------------------------------------
# Environment variables (must be set in HF Space or local run)
# ----------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
TASK_NAME = os.getenv("TASK_NAME", "task1")          # task1, task2, or task3
BENCHMARK = os.getenv("BENCHMARK", "codereview-env")
MAX_STEPS = 1   # single-step environment
TEMPERATURE = 0.0   # deterministic for reproducibility
MAX_TOKENS = 200

# ----------------------------------------------------------------------
# Helper: build system prompt for the LLM
# ----------------------------------------------------------------------
def build_prompt(observation: CodeReviewObservation) -> str:
    if observation.task == "task1":
        return f"""You are a code reviewer. Classify the risk of the following change as LOW, MEDIUM, HIGH, or CRITICAL.

Diff:
{observation.diff}

Output exactly one word: LOW, MEDIUM, HIGH, or CRITICAL."""
    elif observation.task == "task2":
        return f"""You are a code reviewer. Given the dependency map, list all modules that could be affected by the change.

Dependency map:
{json.dumps(observation.dependency_map, indent=2)}

Output a JSON list of module paths, e.g. ["module1.py", "module2.py"]"""
    else:  # task3
        return f"""You are a code reviewer. Make a full review decision.

Diff:
{observation.diff}

Dependency map:
{json.dumps(observation.dependency_map, indent=2)}

Available reviewers: {observation.available_reviewers}

Output a JSON object with fields:
- risk_level (LOW/MEDIUM/HIGH/CRITICAL)
- affected_modules (list of strings)
- recommended_reviewer (string from available_reviewers)
- merge_decision (APPROVE/BLOCK/REQUEST_CHANGES)

Example: {{"risk_level": "HIGH", "affected_modules": ["auth.py"], "recommended_reviewer": "alice", "merge_decision": "BLOCK"}}"""

# ----------------------------------------------------------------------
# Parse LLM response into action
# ----------------------------------------------------------------------
def parse_action(text: str, task: str, episode_id: str) -> CodeReviewAction:
    text = text.strip()
    if task == "task1":
        risk = text.upper()
        if risk not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            risk = "LOW"
        return CodeReviewAction(
            episode_id=episode_id,
            risk_level=risk,
            affected_modules=[],
            recommended_reviewer="",
            merge_decision=""
        )
    elif task == "task2":
        try:
            modules = json.loads(text)
            if not isinstance(modules, list):
                modules = []
        except:
            modules = []
        return CodeReviewAction(
            episode_id=episode_id,
            risk_level="LOW",
            affected_modules=modules,
            recommended_reviewer="",
            merge_decision=""
        )
    else:  # task3
        try:
            data = json.loads(text)
        except:
            data = {}
        return CodeReviewAction(
            episode_id=episode_id,
            risk_level=data.get("risk_level", "LOW"),
            affected_modules=data.get("affected_modules", []),
            recommended_reviewer=data.get("recommended_reviewer", ""),
            merge_decision=data.get("merge_decision", "REQUEST_CHANGES")
        )

# ----------------------------------------------------------------------
# Main inference loop
# ----------------------------------------------------------------------
async def run_episode(task: str) -> tuple[bool, int, list[float]]:
    rewards = []
    error_msg = None
    steps = 0

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Connect to the environment (local or remote Space)
    base_url = os.getenv("ENV_URL", "http://localhost:7860")
    async with HTTPEnvClient(base_url) as env:
        try:
            # Reset
            obs, episode_id = await env.reset(task)
            steps += 1
            action_str = "reset"
            done = False

            # Single-step environment: take exactly one action
            prompt = build_prompt(obs)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            llm_output = response.choices[0].message.content
            action = parse_action(llm_output, task, episode_id)

            # Step
            result = await env.step(action)
            reward = result.reward
            done = result.done
            rewards.append(reward)

            # Emit [STEP] line
            action_json = json.dumps(action.to_dict())
            print(f"[STEP] step={steps} action={action_json} reward={reward:.2f} done={str(done).lower()} error=null")
            sys.stdout.flush()

        except Exception as e:
            error_msg = str(e)
            done = True
            rewards.append(0.0)
            # Still emit a step line for the failure
            print(f"[STEP] step={steps} action=error reward=0.00 done=true error={error_msg}")
            sys.stdout.flush()

    success = not error_msg and (rewards[-1] > 0 if rewards else False)
    return success, steps, rewards

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
async def main():
    task = TASK_NAME
    # Emit [START] line
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}")
    sys.stdout.flush()

    success, steps, rewards = await run_episode(task)

    # Emit [END] line
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")
    sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())
