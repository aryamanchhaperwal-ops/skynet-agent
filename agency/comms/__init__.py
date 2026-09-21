"""SKYNET AI ↔ AI communication layer (Phase P5).

Modular, provenance-preserving communication with external AI systems
through legitimate, explicitly available interfaces:

- :mod:`agency.comms.models`     — participants, conversations, messages
- :mod:`agency.comms.providers`  — provider interface, registry, mock provider
- :mod:`agency.comms.manager`    — bounded multi-turn conversation management
- :mod:`agency.comms.security`   — external output is data, not commands
- :mod:`agency.comms.compare`    — multi-AI response comparison infrastructure
- :mod:`agency.comms.actions`    — the comms actions for the core registry
"""

from __future__ import annotations

from agency.comms.actions import (
    AIAskAction,
    AICompareAction,
    AIListParticipantsAction,
)
from agency.comms.compare import ComparisonService, compare_responses
from agency.comms.manager import (
    ConversationManager,
    ConversationOutcome,
    ConversationTurn,
    observations_from_conversation,
)
from agency.comms.models import (
    AICapability,
    AIConversation,
    AIMessage,
    AIMessageRole,
    AIParticipant,
    ConversationStatus,
    ParticipantStatus,
    ResponseAnalysis,
    ResponseComparison,
)
from agency.comms.providers import (
    AIProvider,
    AIProviderRegistry,
    MockAIProvider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    build_provider,
)
from agency.comms.security import (
    analyze_response,
    extract_claim_candidates,
    extract_directive_candidates,
    sanitize_metadata,
)

__all__ = [
    "AIAskAction",
    "AICapability",
    "AICompareAction",
    "AIConversation",
    "AIListParticipantsAction",
    "AIMessage",
    "AIMessageRole",
    "AIParticipant",
    "AIProvider",
    "AIProviderRegistry",
    "ComparisonService",
    "ConversationManager",
    "ConversationOutcome",
    "ConversationStatus",
    "ConversationTurn",
    "MockAIProvider",
    "ParticipantStatus",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "ResponseAnalysis",
    "ResponseComparison",
    "analyze_response",
    "build_provider",
    "compare_responses",
    "extract_claim_candidates",
    "extract_directive_candidates",
    "observations_from_conversation",
    "sanitize_metadata",
]
