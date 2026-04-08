from __future__ import annotations

import json

import gradio as gr

from incident_rl.envs.unified_env import UnifiedIncidentTriageEnv
from incident_rl.eval import run_policy
from incident_rl.llm import explain_action
from train import heuristic_policy_easy


def summarize_eval(difficulty: str, episodes: int):
    env = UnifiedIncidentTriageEnv(difficulty=difficulty, rng_seed=42)
    random_result = run_policy(env, lambda _: env.action_space.sample(), episodes=episodes)

    if difficulty == "easy":
        heuristic_result = run_policy(env, heuristic_policy_easy, episodes=episodes)
        best_label = "heuristic (proxy for trained RL in starter)"
        best_result = heuristic_result
    else:
        heuristic_result = run_policy(env, lambda _: 0, episodes=episodes)
        best_label = "simple rule (placeholder)"
        best_result = heuristic_result

    rows = [
        ["Random", round(random_result.avg_reward, 2), round(random_result.std_reward, 2)],
        [best_label, round(best_result.avg_reward, 2), round(best_result.std_reward, 2)],
    ]
    return rows


def one_step_explain(difficulty: str):
    env = UnifiedIncidentTriageEnv(difficulty=difficulty, rng_seed=101)
    obs, _ = env.reset()

    if difficulty == "easy":
        action = heuristic_policy_easy(obs)
    else:
        action = env.action_space.sample()

    _, reward, _, _, info = env.step(action)
    explanation = explain_action(obs, f"action_{action}")
    return json.dumps(obs, indent=2, default=float), f"action={action}, reward={reward:.2f}, info={info}", explanation


with gr.Blocks(title="Incident Triage RL Demo") as demo:
    gr.Markdown("# Incident Triage RL Demo\nSwitch difficulty and compare policies quickly.")
    with gr.Row():
        difficulty = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Difficulty")
        episodes = gr.Slider(minimum=20, maximum=500, value=100, step=10, label="Evaluation Episodes")

    compare_btn = gr.Button("Run Policy Comparison")
    comparison_table = gr.Dataframe(headers=["Policy", "Avg Reward", "Std Dev"], datatype=["str", "number", "number"])

    gr.Markdown("## Single-step Explanation")
    explain_btn = gr.Button("Generate Example Decision")
    state_box = gr.Code(label="Sample State", language="json")
    outcome_box = gr.Textbox(label="Action Outcome")
    explanation_box = gr.Textbox(label="Explanation")

    compare_btn.click(fn=summarize_eval, inputs=[difficulty, episodes], outputs=[comparison_table])
    explain_btn.click(fn=one_step_explain, inputs=[difficulty], outputs=[state_box, outcome_box, explanation_box])


if __name__ == "__main__":
    demo.launch()
