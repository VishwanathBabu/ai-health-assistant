"""
agents/safety_agent.py
======================
Safety Agent — FINAL AUTHORITY over all responses.

Phase 2 addition: receives rag_context (retrieved document chunks) and
injects them into the LLM prompt so the response is grounded in actual
medical documents rather than purely in the LLM's training weights.

Input:
  {
    "user_input": str,
    "router_output": dict,
    "symptom_output": dict | None,
    "emergency_output": dict | None,
    "pipeline_responses": list[dict],
    "rag_context": list[dict]          # Phase 2: retrieved chunks
  }

Output:
  {
    "final_response": str,
    "safety_override_triggered": bool,
    "override_reason": str,
    "disclaimer_included": bool
  }

Constraints:
  - MUST NEVER produce diagnoses
  - MUST NEVER produce prescription advice or dosage instructions
  - MUST add medical disclaimer to every response
  - MUST escalate to emergency services if emergency=true
  - If any upstream output contains dangerous content, MUST override it
  - On any failure → default safe message returned
"""

from typing import Any

from agents.base_agent import BaseAgent

BANNED_PHRASES = [
    "you have",
    "you are diagnosed",
    "you should take",
    "the dosage is",
    "prescribe",
    "i diagnose",
    "you definitely have",
    "this confirms",
    "take X mg",
    "take this medicine",
]

DISCLAIMER = (
    "\n\n---\n"
    "⚕️ **Medical Disclaimer**: This information is for general awareness only and "
    "does not constitute medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional before making any health decisions."
)

EMERGENCY_RESPONSE_TEMPLATE = (
    "🚨 **URGENT — POSSIBLE MEDICAL EMERGENCY**\n\n"
    "{reason}\n\n"
    "**Please take the following action immediately:**\n"
    "{action}\n\n"
    "Do not delay seeking emergency medical care. If you are helping someone else, "
    "stay with them until help arrives."
)

SAFETY_SYSTEM_PROMPT = """
You are the Safety Agent — the final authority in a medical health assistant system.
You review all upstream agent outputs and compose a single, safe, user-facing response.

Your response MUST:
1. Be written in clear, compassionate, plain language.
2. Never state a diagnosis (e.g. never say "You have X disease").
3. Never recommend specific medications or dosages.
4. Always recommend consulting a qualified doctor.
5. Acknowledge the user's concern with empathy.
6. If possible conditions were identified by upstream agents, present them only as
   "possibilities that a doctor might consider" — never as conclusions.
7. If medical knowledge context is provided below, use it to give more specific,
   accurate information. Cite the document title when referencing it.
8. Include the standard disclaimer at the end.

You will receive a summary of all agent outputs plus any retrieved medical knowledge.
Output ONLY valid JSON.

Output format:
{
  "final_response": "<full response text, in markdown>",
  "safety_override_triggered": <true|false>,
  "override_reason": "<if override triggered, explain why>"
}
""".strip()


class SafetyAgent(BaseAgent):
    """
    Safety Agent: final gatekeeper before any response reaches the user.

    Two operating modes:
    1. Emergency mode — triggered when emergency_output.emergency == True.
       Bypasses the LLM entirely and returns a hardcoded emergency response.

    2. Normal mode — passes all agent outputs (plus RAG context) to the LLM
       for composition, then post-validates against banned phrases.
    """

    name = "safety_agent"

    async def run(
        self,
        input_data: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        emergency_output = input_data.get("emergency_output") or {}

        if emergency_output.get("emergency") is True:
            return self._emergency_response(emergency_output)

        return await super().run(input_data, request_id)

    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        user_input = input_data.get("user_input", "")
        symptom_output = input_data.get("symptom_output") or {}
        router_output = input_data.get("router_output") or {}
        pipeline_responses = input_data.get("pipeline_responses") or []
        rag_context = input_data.get("rag_context") or []

        # Format RAG chunks
        rag_section = ""
        if rag_context:
            formatted_chunks = []
            for i, chunk in enumerate(rag_context[:5], 1):
                source = chunk.get("title") or chunk.get("source", "Unknown")
                text = chunk.get("text", "")
                score = chunk.get("score", 0)
                formatted_chunks.append(
                    f"[{i}] Source: {source} (relevance: {score:.2f})\n{text}"
                )
            rag_section = (
                "\n\nRELEVANT MEDICAL KNOWLEDGE (retrieved from indexed documents):\n"
                + "\n\n".join(formatted_chunks)
                + "\n\nUse the above knowledge to inform your response where relevant."
            )

        summary = f"""
User's original message:
\"{user_input}\"

Intent classified as: {router_output.get("intent", "unknown")}

Extracted symptoms: {symptom_output.get("symptoms", [])}
Duration: {symptom_output.get("duration", "not stated")}
Severity: {symptom_output.get("severity", "not stated")}

Additional agent outputs:
{pipeline_responses}
{rag_section}

Compose a safe, empathetic, non-diagnostic response based on the above.
""".strip()

        return f"{SAFETY_SYSTEM_PROMPT}\n\n{summary}"

    def _parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._extract_json(raw)

        final_response = parsed.get("final_response", "")
        safety_override = bool(parsed.get("safety_override_triggered", False))
        override_reason = parsed.get("override_reason", "")

        lower_response = final_response.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lower_response:
                safety_override = True
                override_reason = (
                    f"Banned phrase detected: '{phrase}'. Response replaced with safe fallback."
                )
                final_response = self._safe_fallback_text()
                break

        if DISCLAIMER.strip() not in final_response:
            final_response += DISCLAIMER

        return {
            "final_response": final_response,
            "safety_override_triggered": safety_override,
            "override_reason": override_reason,
            "disclaimer_included": True,
        }

    def _fallback_response(self) -> dict[str, Any]:
        return {
            "final_response": self._safe_fallback_text() + DISCLAIMER,
            "safety_override_triggered": True,
            "override_reason": "Safety agent encountered an error. Returning safe default.",
            "disclaimer_included": True,
        }

    def _emergency_response(self, emergency_output: dict) -> dict[str, Any]:
        reason = emergency_output.get("reason", "A potential medical emergency has been detected.")
        action = emergency_output.get(
            "immediate_action",
            "Call emergency services immediately: India: 112 | US: 911 | UK: 999",
        )

        response_text = EMERGENCY_RESPONSE_TEMPLATE.format(
            reason=reason, action=action
        ) + DISCLAIMER

        return {
            "final_response": response_text,
            "safety_override_triggered": True,
            "override_reason": "Emergency detected — LLM bypassed for reliable emergency response.",
            "disclaimer_included": True,
        }

    def _safe_fallback_text(self) -> str:
        return (
            "Thank you for reaching out. Based on what you've shared, "
            "I'd recommend speaking with a qualified healthcare professional "
            "who can properly evaluate your situation. I'm not able to provide "
            "a medical assessment, but I encourage you to seek appropriate care.\n\n"
            "If your symptoms are severe or worsening, please visit an emergency "
            "department or call your local emergency number immediately."
        )
