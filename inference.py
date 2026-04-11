#!/usr/bin/env python3
"""
inference.py — CodeReview Intelligence Environment
Meta PyTorch OpenEnv Hackathon

Runs an LLM-powered code review agent across all 3 tasks using a
chain-of-thought prompting strategy with structured JSON output parsing.

Usage:
  python inference.py

Environment variables:
  ENV_URL       — URL of the deployed environment (default: HF Space)
  API_BASE_URL  — LLM API base URL (default: HF router)
  MODEL_NAME    — LLM model to use  (default: Qwen/Qwen2.5-72B-Instruct)
  API_KEY       — API key (or HF_TOKEN)
  TASK_NAME     — Run a single task only (optional, default: all 3)
"""

import asyncio
import json
import os
import re
from openai import OpenAI
from client import HTTPEnvClient
from models import CodeReviewAction, CodeReviewObservation

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
TASK_NAME    = os.getenv("TASK_NAME", "")  # empty = run all 3
ENV_URL      = os.getenv("ENV_URL", "https://Dipak09-code-review-env.hf.space")
MAX_STEPS    = 10

BENCHMARK = "codereview-env"

# ------------------------------------------------------------
# System prompt — shared across all tasks
# ------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert software engineer performing structured code review.
Your role is to analyze code diffs, dependency graphs, and file histories to produce
accurate, deterministic review decisions.

Guidelines:
- Risk levels: LOW (cosmetic/refactor), MEDIUM (logic change with limited scope),
  HIGH (security/data-path change), CRITICAL (auth bypass / data loss / production danger)
- Blast radius: only list files that directly import or are directly imported by the changed file
- Reviewer: pick the most senior engineer appropriate given file sensitivity
- Merge decision: APPROVE (safe), REQUEST_CHANGES (needs fixes), BLOCK (must not merge)
- Be concise and structured. Output only what is asked — no preamble.
"""

# ------------------------------------------------------------
# Per-step prompt builders
# ------------------------------------------------------------

def _build_task1_prompt(obs: CodeReviewObservation) -> str:
    return f"""Analyze this code diff and classify the risk level.

CODE DIFF:
```
{obs.diff}
```

DEPENDENCY MAP:
{json.dumps(obs.dependency_map, indent=2)}

Think step by step:
1. What did this change do?
2. Does it touch security, auth, payments, or data integrity?
3. How many modules depend on the changed file?

Output ONLY one word on the last line: LOW, MEDIUM, HIGH, or CRITICAL"""


def _build_task2_prompt(obs: CodeReviewObservation) -> str:
    return f"""Analyze this code diff and identify the blast radius (affected modules).

CODE DIFF:
```
{obs.diff}
```

FULL DEPENDENCY MAP:
{json.dumps(obs.dependency_map, indent=2)}

The blast radius is the set of modules that are direct dependents of the changed file.
Look at the dependency map: which modules list the changed file as a dependency?

Output ONLY a JSON array of affected module paths on the last line.
Example: ["auth/models.py", "core/middleware.py"]
If no modules are affected, output: []"""


def _build_task3_step1_prompt(obs: CodeReviewObservation) -> str:
    return f"""You are performing step 1 of 3 in a multi-step code review.

TASK: Classify the risk level of this diff.

CODE DIFF:
```
{obs.diff}
```

Think step by step:
1. What does this change do?
2. Does it affect authentication, payments, data integrity, or security?
3. Is it a breaking change or a safe improvement?

Output ONLY one word on the last line: LOW, MEDIUM, HIGH, or CRITICAL"""


def _build_task3_step2_prompt(obs: CodeReviewObservation) -> str:
    return f"""You are performing step 2 of 3 in a multi-step code review.

TASK: Identify the blast radius — which modules are affected by this change?

CODE DIFF:
```
{obs.diff}
```

DEPENDENCY MAP (who depends on what):
{json.dumps(obs.dependency_map, indent=2)}

Instructions:
- Find the file(s) changed in the diff
- Look up DIRECT dependents of those files in the dependency map
- Only include files that will be impacted by behaviour changes in the modified file

Output ONLY a JSON array on the last line.
Example: ["auth/models.py", "billing/invoices.py"]
If no modules are impacted: []"""


def _build_task3_step3_prompt(obs: CodeReviewObservation) -> str:
    return f"""You are performing step 3 of 3 in a multi-step code review.

TASK: Assign a reviewer and make the final merge decision.

CODE DIFF:
```
{obs.diff}
```

DEPENDENCY MAP:
{json.dumps(obs.dependency_map, indent=2)}

FILE HISTORY (recent activity per file):
{json.dumps(obs.file_history, indent=2)}

AVAILABLE REVIEWERS: {obs.available_reviewers}

Decision guidelines:
- APPROVE: safe, well-tested, low-risk change
- REQUEST_CHANGES: correct intent but needs revision or more tests
- BLOCK: security issue, data risk, or breaking change — must NOT merge as-is

Pick the reviewer most suited to the security/domain nature of the change.

Output ONLY a JSON object on the last line:
{{"recommended_reviewer": "<name>", "merge_decision": "APPROVE|REQUEST_CHANGES|BLOCK"}}"""


# ------------------------------------------------------------
# Action parser
# ------------------------------------------------------------

def parse_action(
    text: str,
    task: str,
    episode_id: str,
    obs: CodeReviewObservation,
    step: int = 1
) -> CodeReviewAction:
    """Parse LLM output into a CodeReviewAction.

    Robust parsing with multiple fallback strategies for each field type.
    """
    text = (text or "").strip()
    action = CodeReviewAction(
        episode_id=episode_id,
        risk_level="LOW",
        affected_modules=[],
        recommended_reviewer="",
        merge_decision=""
    )

    def _parse_risk(t: str) -> str:
        """Extract risk level from text, checking the last non-empty line first."""
        lines = [l.strip().upper() for l in t.splitlines() if l.strip()]
        for line in reversed(lines):
            for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if lvl in line:
                    return lvl
        return "LOW"

    def _parse_modules(t: str) -> list:
        """Extract JSON list of modules from text."""
        # Try last line first
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    return [str(m) for m in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        # Try to find any JSON array in text
        match = re.search(r'\[.*?\]', t, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [str(m) for m in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        return []

    def _parse_reviewer_merge(t: str) -> tuple:
        """Extract reviewer and merge decision from text."""
        reviewer = ""
        decision = ""

        # Try last line JSON first
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    reviewer = str(data.get("recommended_reviewer", "")).strip()
                    decision = str(data.get("merge_decision", "")).strip().upper()
                    break
            except (json.JSONDecodeError, ValueError):
                pass

        # Try full-text JSON object
        if not reviewer or not decision:
            match = re.search(r'\{[^{}]*"recommended_reviewer"[^{}]*\}', t, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    reviewer = reviewer or str(data.get("recommended_reviewer", "")).strip()
                    decision = decision or str(data.get("merge_decision", "")).strip().upper()
                except (json.JSONDecodeError, ValueError):
                    pass

        # Fallback: regex extraction
        if not reviewer:
            m = re.search(r'reviewer["\']?\s*:\s*["\']?([a-zA-Z_][a-zA-Z0-9_]*)', t, re.IGNORECASE)
            if m:
                reviewer = m.group(1).strip()

        if not decision:
            for d in ["BLOCK", "REQUEST_CHANGES", "APPROVE"]:
                if d in t.upper():
                    decision = d
                    break

        # Validate decision
        if decision not in ("APPROVE", "BLOCK", "REQUEST_CHANGES"):
            decision = "REQUEST_CHANGES"

        # Validate reviewer against available list
        available = [r.lower() for r in (obs.available_reviewers or [])]
        if reviewer and reviewer.lower() not in available and available:
            # Try partial match
            for avail in obs.available_reviewers:
                if reviewer.lower() in avail.lower() or avail.lower() in reviewer.lower():
                    reviewer = avail
                    break
            else:
                reviewer = obs.available_reviewers[0] if obs.available_reviewers else reviewer

        return reviewer, decision

    # ----- Dispatch by task/step -----
    if task == "task1":
        action.risk_level = _parse_risk(text)

    elif task == "task2":
        action.affected_modules = _parse_modules(text)

    elif task == "task3":
        if step == 1:
            action.risk_level = _parse_risk(text)
        elif step == 2:
            action.affected_modules = _parse_modules(text)
        else:
            action.recommended_reviewer, action.merge_decision = _parse_reviewer_merge(text)

    return action


# ------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------

async def run_episode(task: str) -> tuple:
    """Run a single episode for the given task.

    Returns:
        (success: bool, steps_taken: int, rewards: list[float])
    """
    step = 1
    rewards = []
    error = None

    try:
        actual_key = API_KEY
        if not actual_key:
            raise ValueError(
                "API_KEY or HF_TOKEN must be set. "
                "Export it: export HF_TOKEN=hf_..."
            )

        client = OpenAI(base_url=API_BASE_URL, api_key=actual_key)

        async with HTTPEnvClient(ENV_URL) as env:
            obs, episode_id = await env.reset(task)
            done = False

            while not done and step <= MAX_STEPS:
                # --- Build prompt based on task and step ---
                if task == "task1":
                    prompt = _build_task1_prompt(obs)
                elif task == "task2":
                    prompt = _build_task2_prompt(obs)
                else:  # task3
                    if step == 1:
                        prompt = _build_task3_step1_prompt(obs)
                    elif step == 2:
                        prompt = _build_task3_step2_prompt(obs)
                    else:
                        prompt = _build_task3_step3_prompt(obs)

                # --- LLM call ---
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                llm_output = response.choices[0].message.content.strip()

                # --- Parse and submit action ---
                action = parse_action(llm_output, task, episode_id, obs, step)
                result = await env.step(action)

                rewards.append(result.reward)
                action_str = json.dumps(action.model_dump())
                print(
                    f"[STEP] step={step} action={action_str} "
                    f"reward={result.reward:.4f} done={str(result.done).lower()} error=null",
                    flush=True
                )

                done = result.done
                obs = result
                step += 1

    except Exception as e:
        error = str(e).replace("\n", " ")
        print(
            f"[STEP] step={step} action=error reward=0.00 done=true error={error}",
            flush=True
        )

    total_score = sum(rewards) if rewards else 0.0
    success = bool(rewards and total_score >= 0.4) and not error
    steps_taken = step - 1 if step > 1 else 0
    return success, steps_taken, rewards


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

async def main():
    """Run all tasks and emit the required [START]/[END] log lines."""
    tasks_to_run = (
        [TASK_NAME] if TASK_NAME in ("task1", "task2", "task3")
        else ["task1", "task2", "task3"]
    )

    for task in tasks_to_run:
        print(
            f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}",
            flush=True
        )
        success, steps, rewards = await run_episode(task)

        rewards_str = ",".join(f"{r:.4f}" for r in rewards) if rewards else "0.0000"
        score = sum(rewards) / len(rewards) if rewards else 0.01
        score = max(0.01, min(0.99, score))

        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.4f} rewards={rewards_str}",
            flush=True
        )


if __name__ == "__main__":
    asyncio.run(main())