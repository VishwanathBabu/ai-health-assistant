"""
core/logger.py
==============
Structured, JSON-capable logging for observability.

Every agent call logs:
  - request_id
  - agent name
  - execution time (ms)
  - safety overrides triggered
  - retrieved documents (Phase 2)
  - any exceptions
"""

import time
import uuid

import structlog

from core.config import settings, LogFormat


def _configure_structlog() -> None:
    """Configure structlog once at import time."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.log_format == LogFormat.JSON:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_structlog()
log = structlog.get_logger()


class AgentLogger:
    """
    Context-manager that times an agent call and emits a structured log on exit.

    Usage:
        with AgentLogger("symptom_agent", request_id=rid) as agent_log:
            result = agent.run(input)
            agent_log.set_result(result)
    """

    def __init__(self, agent_name: str, request_id: str | None = None) -> None:
        self.agent_name = agent_name
        self.request_id = request_id or str(uuid.uuid4())
        self._start: float = 0.0
        self._result: dict = {}
        self._safety_override: bool = False
        self._error: Exception | None = None

    def __enter__(self) -> "AgentLogger":
        self._start = time.perf_counter()
        structlog.contextvars.bind_contextvars(
            request_id=self.request_id,
            agent=self.agent_name,
        )
        log.info("agent_start")
        return self

    def set_result(self, result: dict) -> None:
        self._result = result

    def set_safety_override(self, triggered: bool) -> None:
        self._safety_override = triggered

    def set_error(self, error: Exception) -> None:
        self._error = error

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)

        if exc_val is not None:
            log.error(
                "agent_error",
                elapsed_ms=elapsed_ms,
                error=str(exc_val),
                error_type=type(exc_val).__name__,
            )
        else:
            log.info(
                "agent_complete",
                elapsed_ms=elapsed_ms,
                safety_override=self._safety_override,
                result_keys=list(self._result.keys()) if self._result else [],
            )

        structlog.contextvars.clear_contextvars()
        return False  # Never suppress exceptions


def get_request_id() -> str:
    """Generate a new unique request ID."""
    return str(uuid.uuid4())
