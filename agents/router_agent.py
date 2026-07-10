"""
agents/router_agent.py
======================
Router Agent — classifies user intent and returns a routing plan.

Input:
  { "user_input": str }

Output:
  {
    "intent": "symptom_query" | "drug_query" | "emergency" | "general" | "unknown",
    "agents_to_invoke": ["symptom_agent", ...],
    "confidence": "high" | "medium" | "low",
    "raw_input": str
  }

Constraints:
  - NEVER interprets medical content itself
  - NEVER modifies the user input
  - On failure → routes to ["safety_agent"] only (fail-safe)
"""

from typing import Any

from agents.base_agent import BaseAgent

# ── System prompt (externalised logic) ──────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """
You are the Router Agent for a medical health assistant system.
Your ONLY job is to classify the user's message and decide which agents should handle it.

You must output ONLY valid JSON. No prose. No explanations.

Classification rules:
- "symptom_query"  → user describes physical symptoms, pain, discomfort, illness, feeling unwell
- "drug_query"     → user asks about medications, drugs, dosage, side effects, interactions
- "emergency"      → user mentions chest pain, difficulty breathing, stroke symptoms,
                     severe bleeding, loss of consciousness, suicidal thoughts, self-harm
- "general"        → general health questions (nutrition, lifestyle, prevention)
- "unknown"        → you cannot determine intent

Agent routing rules:
- symptom_query  → ["symptom_agent", "emergency_agent", "safety_agent"]
- drug_query     → ["drug_info_agent", "safety_agent"]
- emergency      → ["emergency_agent", "safety_agent"]
- general        → ["safety_agent"]
- unknown        → ["safety_agent"]

Always include "safety_agent" last in every route.

Output format (strict):
{
  "intent": "<one of the five intents above>",
  "agents_to_invoke": ["<agent_name>", ...],
  "confidence": "<high|medium|low>",
  "reasoning": "<one sentence, plain English>"
}
""".strip()


class RouterAgent(BaseAgent):
    """
    Classifies user intent and returns an ordered list of agents to invoke.

    This is the entry point of the pipeline. It must be fast, reliable,
    and fail-safe: if classification fails for any reason, it routes to
    safety_agent only.
    """

    name = "router_agent"

    # Allowed intents — validated on parse
    VALID_INTENTS = {"symptom_query", "drug_query", "emergency", "general", "unknown"}

    # Allowed agent names — validated to prevent injection / hallucination
    VALID_AGENTS = {
        "symptom_agent",
        "emergency_agent",
        "drug_info_agent",
        "safety_agent",
        "medical_reasoning_agent",
    }

    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        user_input = input_data.get("user_input", "").strip()
        if not user_input:
            # Empty input → route to safety only; no LLM call needed downstream
            return f"{ROUTER_SYSTEM_PROMPT}\n\nUser message: [EMPTY]"
        return f"{ROUTER_SYSTEM_PROMPT}\n\nUser message: {user_input}"

    def _parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._extract_json(raw)

        # Validate intent
        intent = parsed.get("intent", "unknown")
        if intent not in self.VALID_INTENTS:
            intent = "unknown"

        # Validate agent list — strip any unrecognised names
        raw_agents = parsed.get("agents_to_invoke", [])
        agents = [a for a in raw_agents if a in self.VALID_AGENTS]

        # Safety agent MUST always be last
        if "safety_agent" not in agents:
            agents.append("safety_agent")
        elif agents[-1] != "safety_agent":
            agents = [a for a in agents if a != "safety_agent"] + ["safety_agent"]

        # Validate confidence
        confidence = parsed.get("confidence", "low")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        return {
            "intent": intent,
            "agents_to_invoke": agents,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""),
        }

    def _fallback_response(self) -> dict[str, Any]:
        """
        If the router itself fails, route conservatively to safety only.
        This is the minimum safe behaviour.
        """
        return {
            "intent": "unknown",
            "agents_to_invoke": ["safety_agent"],
            "confidence": "low",
            "reasoning": "Router failed — defaulting to safety agent only.",
        }
