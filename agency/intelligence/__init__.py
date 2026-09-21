"""Intelligence layer for Skynet (Phase P2).

Public surface: :class:`IntelligenceService` (the single doorway to model
reasoning), the provider abstraction and implementations in
:mod:`agency.intelligence.providers`, and the LLM planner/evaluator
wrappers with deterministic fallback.
"""

from agency.intelligence.evaluator import LLMEvaluator
from agency.intelligence.planner import LLMPlanner
from agency.intelligence.providers import (
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    build_llm_provider,
)
from agency.intelligence.service import IntelligenceService

__all__ = [
    "IntelligenceService",
    "LLMEvaluator",
    "LLMPlanner",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "build_llm_provider",
]
