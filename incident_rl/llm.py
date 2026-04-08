from __future__ import annotations

import os
from typing import Any


def _openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def explain_action(state: dict[str, Any], action_name: str) -> str:
    """Generate a short analyst-style rationale for a chosen action."""
    client = _openai_client()
    if client is None:
        return (
            f"Heuristic explanation: action '{action_name}' was selected from alert/service context "
            "to balance risk, operational cost, and response speed."
        )

    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    prompt = (
        "You are a senior incident analyst. Explain in 2-3 concise bullet points why this action "
        f"is reasonable.\nState: {state}\nAction: {action_name}"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are concise, practical, and security-focused."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No explanation generated."
    except Exception as exc:
        return f"LLM explanation unavailable ({exc})."


def llm_baseline_policy(state: dict[str, Any], n_actions: int) -> int:
    """Optional LLM baseline policy. Falls back to a conservative heuristic."""
    client = _openai_client()
    if client is None:
        # Conservative fallback heuristic: choose middle-safe action where possible.
        return min(2, n_actions - 1)

    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    prompt = (
        f"Choose one integer action in range [0, {n_actions - 1}] for this incident state. "
        "Output only the integer."
        f"\nState: {state}"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a careful incident responder."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        text = (response.choices[0].message.content or "").strip()
        action = int("".join(c for c in text if c.isdigit()) or "0")
        return max(0, min(n_actions - 1, action))
    except Exception:
        return min(2, n_actions - 1)
