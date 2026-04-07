#!/usr/bin/env python3
import asyncio
import json
import os
import re
from openai import OpenAI
from client import HTTPEnvClient
from models import CodeReviewAction, CodeReviewObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")  # validator injects API_KEY
TASK_NAME    = os.getenv("TASK_NAME", "task3")
ENV_URL      = os.getenv("ENV_URL", "https://Dipak09-code-review-env.hf.space")
MAX_STEPS = 10

async def run_episode():
    print(f"[START] task={TASK_NAME} env=codereview-env model={MODEL_NAME}", flush=True)


    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    async with HTTPEnvClient(ENV_URL) as env:
        obs, episode_id = await env.reset(TASK_NAME)
        step = 1
        rewards = []
        done = False
        error = None

        try:
            while not done and step <= MAX_STEPS:
                # Build prompt based on observation content
                if TASK_NAME == "task3":
                    if step == 1:
                        # Step 1: only risk level needed
                        prompt = f"Diff:\n{obs.diff}\n\nOutput only the risk level (LOW/MEDIUM/HIGH/CRITICAL)."
                    elif step == 2:
                        # Step 2: only blast radius needed
                        prompt = f"Diff:\n{obs.diff}\n\nDependency map:\n{json.dumps(obs.dependency_map, indent=2)}\n\nOutput a JSON list of affected modules, e.g., [\"auth.py\"]."
                    else:
                        # Step 3: full decision
                        prompt = f"Diff:\n{obs.diff}\n\nDependency map:\n{json.dumps(obs.dependency_map, indent=2)}\n\nFile history:\n{json.dumps(obs.file_history, indent=2)}\n\nAvailable reviewers: {obs.available_reviewers}\n\nOutput a JSON object with fields: recommended_reviewer (string) and merge_decision (APPROVE/BLOCK/REQUEST_CHANGES)."
                elif TASK_NAME == "task1":
                    prompt = f"Diff:\n{obs.diff}\n\nOutput only the risk level (LOW/MEDIUM/HIGH/CRITICAL)."
                else:  # task2
                    prompt = f"Diff:\n{obs.diff}\n\nDependency map:\n{json.dumps(obs.dependency_map, indent=2)}\n\nOutput a JSON list of affected modules, e.g., [\"auth.py\"]."

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                llm_output = response.choices[0].message.content.strip()
                action = parse_action(llm_output, TASK_NAME, episode_id, obs, step)

                result = await env.step(action)
                rewards.append(result.reward)
                action_str = json.dumps(action.model_dump())  # FIX: .model_dump() for Pydantic v2
                print(f"[STEP] step={step} action={action_str} reward={result.reward:.2f} done={str(result.done).lower()} error=null", flush=True)
                done = result.done
                obs = result   # FIX: result IS the observation, not result.observation
                step += 1

        except Exception as e:
            error = str(e)
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={error}", flush=True)

        total_score = sum(rewards) if rewards else 0.0
        success = bool(rewards and total_score >= 0.5) if not error else False
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
        print(f"[END] success={str(success).lower()} steps={step-1} score={total_score:.3f} rewards={rewards_str}", flush=True)

def parse_action(text: str, task: str, episode_id: str, obs: CodeReviewObservation, step: int = 1) -> CodeReviewAction:
    text = text.strip()
    action = CodeReviewAction(episode_id=episode_id, risk_level="LOW", affected_modules=[], recommended_reviewer="", merge_decision="")

    if task == "task1":
        risk = text.upper()
        if risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            action.risk_level = risk
    elif task == "task2":
        try:
            modules = json.loads(text)
            if isinstance(modules, list):
                action.affected_modules = modules
        except:
            match = re.search(r'\[.*?\]', text)
            if match:
                try:
                    modules = json.loads(match.group())
                    if isinstance(modules, list):
                        action.affected_modules = modules
                except:
                    pass
    else:  # task3
        if step == 1:
            risk = text.upper()
            if risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                action.risk_level = risk
        elif step == 2:
            try:
                modules = json.loads(text)
                if isinstance(modules, list):
                    action.affected_modules = modules
            except:
                match = re.search(r'\[.*?\]', text)
                if match:
                    try:
                        modules = json.loads(match.group())
                        if isinstance(modules, list):
                            action.affected_modules = modules
                    except:
                        pass
        else:
            try:
                match = re.search(r'\{.*?\}', text, re.DOTALL)
                data = json.loads(match.group()) if match else {}
                if "recommended_reviewer" in data:
                    action.recommended_reviewer = data["recommended_reviewer"]
                if "merge_decision" in data:
                    decision = data["merge_decision"].upper()
                    if decision in ["APPROVE", "BLOCK", "REQUEST_CHANGES"]:
                        action.merge_decision = decision
            except:
                if "reviewer" in text.lower():
                    match = re.search(r'reviewer["\']?\s*:\s*["\']?([a-zA-Z0-9_]+)', text, re.IGNORECASE)
                    if match:
                        action.recommended_reviewer = match.group(1)
                if "merge" in text.lower():
                    if "approve" in text.lower():
                        action.merge_decision = "APPROVE"
                    elif "block" in text.lower():
                        action.merge_decision = "BLOCK"
                    else:
                        action.merge_decision = "REQUEST_CHANGES"
    return action

if __name__ == "__main__":
    asyncio.run(run_episode())