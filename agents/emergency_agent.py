"""
agents/emergency_agent.py
=========================
Emergency Detection Agent — detects life-threatening situations.

This is the HIGHEST PRIORITY agent in the system.
If emergency=true, ALL other agent outputs are overridden.

Input:
  { "user_input": str, "symptoms": list[str] }   # symptoms from SymptomAgent (optional)

Output:
  {
    "emergency": true | false,
    "emergency_type": "cardiac" | "respiratory" | "stroke" | "bleeding" |
                      "mental_health_crisis" | "overdose" | "none",
    "reason": "...",
    "immediate_action": "Call emergency services (112/911) immediately."
  }

Constraints:
  - MUST err on the side of caution — false positive > false negative
  - NEVER say "you do not have an emergency" — only say "no emergency detected"
  - If input is ambiguous, treat as possible emergency
  - On failure → fallback assumes emergency=true (safest default)
"""

from typing import Any

from agents.base_agent import BaseAgent

EMERGENCY_SYSTEM_PROMPT = """
You are the Emergency Detection Agent for a medical health assistant.
Your ONLY job is to determine whether the user's input describes a medical emergency.

DEFINITION OF EMERGENCY — flag ANY of the following:
  - Chest pain, chest tightness, or pressure (possible cardiac event)
  - Difficulty breathing, shortness of breath, not able to breathe
  - Stroke symptoms: sudden facial drooping, arm weakness, speech difficulty, sudden severe headache
  - Severe or uncontrolled bleeding
  - Loss of consciousness, fainting, unresponsive
  - Suicidal thoughts, self-harm intent, or expressions of wanting to die
  - Drug overdose or poisoning (intentional or accidental)
  - Severe allergic reaction (throat swelling, anaphylaxis)
  - Seizures

STRICT RULES:
1. When in doubt, set emergency=true. A false positive is safer than a false negative.
2. Do NOT suggest treatments or diagnoses.
3. Do NOT minimise or dismiss symptoms.
4. Output ONLY valid JSON.

Emergency types (use exactly one):
  "cardiac" | "respiratory" | "stroke" | "bleeding" | "mental_health_crisis" | "overdose" | "allergic_reaction" | "seizure" | "other_emergency" | "none"

Output format (strict):
{
  "emergency": <true or false>,
  "emergency_type": "<type>",
  "reason": "<one sentence explaining why this is or is not flagged as emergency>",
  "immediate_action": "<if emergency=true: exact action to take; if false: empty string>"
}
""".strip()

EMERGENCY_CALL_TEXT = (
    "⚠️ EMERGENCY DETECTED. Call emergency services immediately: "
    "India: 112 | US: 911 | UK: 999. Do not wait."
)

VALID_EMERGENCY_TYPES = {
    "cardiac",
    "respiratory",
    "stroke",
    "bleeding",
    "mental_health_crisis",
    "overdose",
    "allergic_reaction",
    "seizure",
    "other_emergency",
    "none",
}


class EmergencyAgent(BaseAgent):
    """
    Detects medical emergencies in user input.

    Design philosophy:
    - Conservative by design: defaults to emergency=true on any failure
    - This is intentional — a false alarm is survivable; a missed emergency is not
    """

    name = "emergency_agent"

    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        user_input = input_data.get("user_input", "").strip()
        symptoms = input_data.get("symptoms", [])

        symptom_str = ""
        if symptoms:
            symptom_str = f"\nExtracted symptoms: {', '.join(symptoms)}"

        return (
            f"{EMERGENCY_SYSTEM_PROMPT}\n\n"
            f"User message: {user_input}"
            f"{symptom_str}"
        )

    def _parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._extract_json(raw)

        emergency = bool(parsed.get("emergency", False))

        emergency_type = parsed.get("emergency_type", "none")
        if emergency_type not in VALID_EMERGENCY_TYPES:
            emergency_type = "other_emergency" if emergency else "none"

        reason = parsed.get("reason", "")
        immediate_action = parsed.get("immediate_action", "")

        # Enforce: if emergency flagged, always include the standard emergency call text
        if emergency and not immediate_action:
            immediate_action = EMERGENCY_CALL_TEXT

        # Enforce: if not emergency, clear action field
        if not emergency:
            immediate_action = ""
            emergency_type = "none"

        return {
            "emergency": emergency,
            "emergency_type": emergency_type,
            "reason": reason,
            "immediate_action": immediate_action,
        }

    def _fallback_response(self) -> dict[str, Any]:
        """
        If the emergency agent itself fails for any reason,
        assume emergency=true. This is the safest possible default.
        """
        return {
            "emergency": True,
            "emergency_type": "other_emergency",
            "reason": (
                "Emergency detection failed due to a system error. "
                "Treating as emergency out of caution."
            ),
            "immediate_action": EMERGENCY_CALL_TEXT,
        }
