#!/usr/bin/env python3
import asyncio
import os
import json
from openai import OpenAI
from client import HTTPEnvClient
from models import CodeReviewAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TASK_NAME = os.getenv("TASK_NAME", "task3")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS = 10  # safety

def build_prompt(obs):
    # Customize per task and step  simple version
    if obs.task == "task3":
        if obs.dependency_map and not obs.file_history:
            return f"Diff:\n{obs.diff}\n\nDependency map:\n{json.dumps(obs.dependency_map, indent=2)}\n\nOutput affected modules as JSON list."
        elif obs.file_history:
            return f"Diff:\n{obs.diff}\n\nDependency map:\n{json.dumps(obs.dependency_map, indent=2)}\n\nHistory:\n{json.dumps(obs.file_history, indent=2)}\n\nAvailable reviewers: {obs.available_reviewers}\n\nOutput JSON with recommended_reviewer and merge_decision."
        else:
            return f"Diff:\n{obs.diff}\n\nOutput risk level (LOW/MEDIUM/HIGH/CRITICAL) in JSON: {{\"risk_level\": \"...\"}}."
    return f"Diff:\n{obs.diff}\nProvide relevant output in JSON."

def parse_action(text, task, episode_id, step):
    try:
        data = json.loads(text)
    except:
        data = {}
    return CodeReviewAction(
        episode_id=episode_id,
        risk_level=data.get("risk_level", "LOW"),
        affected_modules=data.get("affected_modules", []) if isinstance(data.get("affected_modules", []), list) else [],
        recommended_reviewer=data.get("recommended_reviewer", ""),
        merge_decision=data.get("merge_decision", "")
    )

async def run_episode():
    print(f"[START] task={TASK_NAME} env=codereview-env model={MODEL_NAME}", flush=True)
    if not HF_TOKEN:
        print("[STEP] step=1 action=error reward=0.00 done=true error=HF_TOKEN_MISSING", flush=True)
        print("[END] success=false steps=0 rewards=0.00", flush=True)
        return
        
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    async with HTTPEnvClient(ENV_URL) as env:
        obs, episode_id = await env.reset(TASK_NAME)
        step = 1
        rewards = []
        error = None
        done = False
        try:
            while not done and step <= MAX_STEPS:
                prompt = build_prompt(obs)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200
                )
                action_text = response.choices[0].message.content
                action = parse_action(action_text, TASK_NAME, episode_id, step)
                result = await env.step(action)
                rewards.append(result.reward)
                action_str = json.dumps(action.to_dict())
                print(f"[STEP] step={step} action={action_str} reward={result.reward:.2f} done={str(result.done).lower()} error=null", flush=True)
                done = result.done
                obs = result.observation
                step += 1
        except Exception as e:
            error = str(e)
            print(f"[STEP] step={step} action=error reward=0.00 done=true error={error}", flush=True)
        finally:
            total_reward = sum(rewards)
            # The sample script uses a normalized score in [0, 1]
            # Max possible reward for task3 is 1.0 (0.25+0.3+0.45)
            # For simplicity in this environment, score = total_reward if already capped at 1.0
            score = min(max(total_reward, 0.0), 1.0)
            success = score >= 0.5
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={str(success).lower()} steps={step-1} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(run_episode())
