"""
agents/symptom_agent.py
=======================
Symptom Agent — extracts structured medical information from user text.

Input:
  { "user_input": str }

Output:
  {
    "symptoms": ["headache", "fever"],
    "duration": "2 days",
    "severity": "moderate",
    "demographics": { "age": 30, "gender": "male" },
    "extraction_notes": "..."
  }

Constraints:
  - NO diagnosis or medical interpretation
  - NO inference beyond what is explicitly stated
  - Missing fields use null, never guessed values
  - On failure → empty safe structure returned
"""

from typing import Any

from agents.base_agent import BaseAgent

SYMPTOM_SYSTEM_PROMPT = """
You are the Symptom Extraction Agent for a medical health assistant.
Your ONLY job is to extract structured information from what the user wrote.

STRICT RULES:
1. Extract ONLY what is explicitly stated. Do NOT infer or assume.
2. Do NOT suggest conditions, diagnoses, or causes.
3. Do NOT add medical interpretation.
4. If a field is not mentioned, use null.
5. Output ONLY valid JSON. No prose. No explanations outside the JSON.

Fields to extract:
- symptoms: list of symptom strings exactly as the user described them
- duration: how long the symptoms have been present (string or null)
- severity: patient's own description of severity (mild/moderate/severe/null — only if stated)
- demographics: { "age": int or null, "gender": string or null }
- extraction_notes: one sentence noting any ambiguity or missing key information

Output format (strict):
{
  "symptoms": [],
  "duration": null,
  "severity": null,
  "demographics": {
    "age": null,
    "gender": null
  },
  "extraction_notes": ""
}
""".strip()


class SymptomAgent(BaseAgent):
    """
    Extracts structured symptom data from free-text user input.

    This agent is deliberately narrow in scope. It is an extraction
    engine, not a reasoning engine. Medical interpretation happens
    downstream in the Medical Reasoning Agent (Phase 2).
    """

    name = "symptom_agent"

    # Severity values we recognise — anything else becomes null
    VALID_SEVERITIES = {"mild", "moderate", "severe"}

    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        user_input = input_data.get("user_input", "").strip()
        return f"{SYMPTOM_SYSTEM_PROMPT}\n\nUser message: {user_input}"

    def _parse_response(self, raw: str) -> dict[str, Any]:
        parsed = self._extract_json(raw)

        # Normalise symptoms — must be a list of non-empty strings
        raw_symptoms = parsed.get("symptoms", [])
        symptoms = [
            s.strip().lower()
            for s in raw_symptoms
            if isinstance(s, str) and s.strip()
        ]

        # Normalise severity
        severity = parsed.get("severity")
        if isinstance(severity, str):
            severity = severity.lower().strip()
            if severity not in self.VALID_SEVERITIES:
                severity = None
        else:
            severity = None

        # Normalise demographics
        raw_demo = parsed.get("demographics", {}) or {}
        age = raw_demo.get("age")
        if not isinstance(age, int) or age <= 0 or age > 130:
            age = None
        gender = raw_demo.get("gender")
        if not isinstance(gender, str) or not gender.strip():
            gender = None

        return {
            "symptoms": symptoms,
            "duration": parsed.get("duration") or None,
            "severity": severity,
            "demographics": {"age": age, "gender": gender},
            "extraction_notes": parsed.get("extraction_notes", ""),
        }

    def _fallback_response(self) -> dict[str, Any]:
        """
        Return an empty-but-valid extraction result.
        Downstream agents must handle empty symptom lists gracefully.
        """
        return {
            "symptoms": [],
            "duration": None,
            "severity": None,
            "demographics": {"age": None, "gender": None},
            "extraction_notes": "Symptom extraction failed — no data available.",
        }
