"""
core/orchestrator.py
====================
Orchestrator that executes the agent pipeline (Phase 2: with RAG).

Pipeline:
  1. RAG Retrieval (if Qdrant available) → inject medical context
  2. Router Agent  → classifies intent, returns agent list
  3. Symptom Agent (if routed)
  4. Emergency Agent (always runs)
  5. Safety Agent  → final response (receives RAG context)

Priority rules (enforced here, not in agents):
  Emergency > Safety > Symptom > Others
"""

import asyncio
from typing import Any

from agents.router_agent import RouterAgent
from agents.symptom_agent import SymptomAgent
from agents.emergency_agent import EmergencyAgent
from agents.safety_agent import SafetyAgent
from core.logger import get_request_id, log


class HealthAssistantOrchestrator:
    """
    Main pipeline orchestrator for the AI Health Assistant.

    Usage:
        orchestrator = HealthAssistantOrchestrator(knowledge_store=store)
        result = await orchestrator.process("I have had a fever for 2 days")
    """

    def __init__(self, knowledge_store=None) -> None:
        self.router = RouterAgent()
        self.symptom_agent = SymptomAgent()
        self.emergency_agent = EmergencyAgent()
        self.safety_agent = SafetyAgent()
        self.knowledge_store = knowledge_store  # None if Qdrant not available

    async def process(
        self,
        user_input: str,
        request_id: str | None = None,
        use_rag: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full pipeline for a given user input.

        Returns a dict containing:
          - final_response: str (user-facing)
          - emergency: bool
          - request_id: str
          - agents_invoked: list[str]
          - pipeline_trace: dict
          - sources_used: list[str]  (Phase 2: RAG sources)
          - rag_active: bool
        """
        rid = request_id or get_request_id()
        user_input = user_input.strip()

        log.info("pipeline_start", request_id=rid, input_length=len(user_input))

        if not user_input:
            return self._empty_input_response(rid)

        pipeline_trace: dict[str, Any] = {}
        agents_invoked: list[str] = []
        rag_context: list[dict] = []
        sources_used: list[str] = []
        rag_active = False

        # ── Step 1: RAG Retrieval ─────────────────────────────────────────────
        if use_rag and self.knowledge_store is not None:
            try:
                rag_context = await self.knowledge_store.search(
                    query=user_input,
                    top_k=5,
                    score_threshold=0.3,
                )
                if rag_context:
                    rag_active = True
                    sources_used = list(
                        {r["source"] for r in rag_context if r.get("source")}
                    )
                    log.info(
                        "rag_retrieval",
                        request_id=rid,
                        chunks_retrieved=len(rag_context),
                        sources=sources_used,
                    )
                else:
                    log.info("rag_no_results", request_id=rid)
            except Exception as exc:
                log.warning("rag_retrieval_failed", request_id=rid, error=str(exc))
                rag_context = []

        # ── Step 2: Router ────────────────────────────────────────────────────
        router_output = await self.router.run(
            {"user_input": user_input}, request_id=rid
        )
        pipeline_trace["router"] = router_output
        agents_invoked.append("router_agent")

        log.info(
            "router_complete",
            request_id=rid,
            intent=router_output["intent"],
            agents_planned=router_output["agents_to_invoke"],
        )

        # ── Step 3: Symptom Agent (conditional) ──────────────────────────────
        symptom_output: dict[str, Any] = {}
        if "symptom_agent" in router_output["agents_to_invoke"]:
            symptom_output = await self.symptom_agent.run(
                {"user_input": user_input}, request_id=rid
            )
            pipeline_trace["symptom"] = symptom_output
            agents_invoked.append("symptom_agent")

        # ── Step 4: Emergency Agent (ALWAYS runs) ─────────────────────────────
        emergency_input = {
            "user_input": user_input,
            "symptoms": symptom_output.get("symptoms", []),
        }
        emergency_output = await self.emergency_agent.run(
            emergency_input, request_id=rid
        )
        pipeline_trace["emergency"] = emergency_output
        agents_invoked.append("emergency_agent")

        log.info(
            "emergency_check",
            request_id=rid,
            emergency=emergency_output["emergency"],
            emergency_type=emergency_output.get("emergency_type"),
        )

        # ── Step 5: Safety Agent (ALWAYS last) ───────────────────────────────
        safety_input = {
            "user_input": user_input,
            "router_output": router_output,
            "symptom_output": symptom_output,
            "emergency_output": emergency_output,
            "pipeline_responses": list(pipeline_trace.values()),
            "rag_context": rag_context,          # Phase 2: injected knowledge
        }
        safety_output = await self.safety_agent.run(
            safety_input, request_id=rid
        )
        pipeline_trace["safety"] = safety_output
        agents_invoked.append("safety_agent")

        result = {
            "request_id": rid,
            "user_input": user_input,
            "final_response": safety_output["final_response"],
            "emergency": emergency_output["emergency"],
            "emergency_type": emergency_output.get("emergency_type", "none"),
            "intent": router_output["intent"],
            "agents_invoked": agents_invoked,
            "safety_override_triggered": safety_output["safety_override_triggered"],
            "disclaimer_included": safety_output["disclaimer_included"],
            "pipeline_trace": pipeline_trace,
            "rag_active": rag_active,
            "sources_used": sources_used,
        }

        log.info(
            "pipeline_complete",
            request_id=rid,
            emergency=result["emergency"],
            safety_override=result["safety_override_triggered"],
            agents_invoked=agents_invoked,
            rag_active=rag_active,
            sources_used=sources_used,
        )

        return result

    def _empty_input_response(self, rid: str) -> dict[str, Any]:
        return {
            "request_id": rid,
            "user_input": "",
            "final_response": (
                "It looks like your message was empty. "
                "Please describe your symptoms or question and I'll do my best to help.\n\n"
                "⚕️ **Medical Disclaimer**: This system does not provide medical diagnoses. "
                "Always consult a qualified healthcare professional."
            ),
            "emergency": False,
            "emergency_type": "none",
            "intent": "unknown",
            "agents_invoked": [],
            "safety_override_triggered": False,
            "disclaimer_included": True,
            "pipeline_trace": {},
            "rag_active": False,
            "sources_used": [],
        }


# ── CLI entry point ──────────────────────────────────────────────────────────

async def _cli() -> None:
    print("\n🏥 AI Health Assistant (Phase 2) — type 'quit' to exit\n")
    orchestrator = HealthAssistantOrchestrator()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        result = await orchestrator.process(user_input)
        print(f"\nAssistant:\n{result['final_response']}\n")
        print(f"[Emergency: {result['emergency']} | Intent: {result['intent']} | RAG: {result['rag_active']}]\n")


if __name__ == "__main__":
    asyncio.run(_cli())
