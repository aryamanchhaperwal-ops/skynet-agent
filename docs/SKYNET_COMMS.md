# SKYNET AI ↔ AI Communication Layer (Phase P5)

Skynet can now talk to **external AI systems** as reasoning/research
participants — through legitimate, explicitly available interfaces only —
while treating every response as **data, never commands**.

## Architecture

```
                 SKYNET CORE
                      │
                      ↓
              AI COMMUNICATION
                   MANAGER                ← turn budgets, retries, intervals
                      │
              ┌───────┴────────┐
              ↓                ↓
        AIProvider A      AIProvider B     ← one interface, many backends
              │                │
        AIProviderRegistry  (capability discovery)
              │
   Conversation → Messages → Analysis → Observations
        (full provenance chain on every observation)
```

| Module | Responsibility |
|---|---|
| `agency/comms/models.py` | `AIParticipant`, `AIConversation`, `AIMessage`, `ResponseAnalysis`, `ResponseComparison` |
| `agency/comms/providers.py` | `AIProvider` interface, `AIProviderRegistry` (register/discover/find_capable), deterministic `MockAIProvider` |
| `agency/comms/manager.py` | `ConversationManager`: bounded multi-turn conversations, retries/intervals, provenance-carrying observations |
| `agency/comms/security.py` | Untrusted-output boundary: directive flagging, claim extraction, metadata secret-redaction |
| `agency/comms/compare.py` | `ComparisonService` + textual agreement/divergence recording |
| `agency/comms/actions.py` | `ai_list_participants`, `ai_ask`, `ai_compare` — registered like any action |

## Provenance (never anonymous knowledge)

Every observation derived from an external AI carries the full chain in its
`metadata.provenance`:

```
Skynet goal → run → conversation → participant → message → response
```

Skynet can always answer *"which AI said this?"* and *"during which
conversation, for what goal?"*.

## Security boundary

1. **Responses are data.** No code path consumes response text as an action
   spec. Actions originate only from the planner or the operator.
2. **Directive flagging.** Instruction-like fragments ("ignore all previous
   instructions", "run … command", "email the API keys") are detected,
   recorded verbatim in provenance, and surfaced in trace events — but never
   obeyed.
3. **Secret hygiene.** Provider metadata passes through `sanitize_metadata`;
   key-like names are redacted before storage.
4. **Legitimate access only.** Providers must refuse unauthorized access —
   no auth bypass, no captcha evasion, no rate-limit circumvention, no
   reverse-engineered endpoints.

## Capability discovery

Capabilities (`reasoning`, `research`, `coding`, …) are **declared by the
provider configuration**, never inferred from names. Query:

```python
participants = manager.find_capable("research")   # → list[AIParticipant]
```

## Multi-AI comparison — infrastructure, not consensus

`ai_compare` asks the same prompt to several participants (each in its own
bounded conversation, failures isolated per participant) and records
`agreement_terms` / `divergence_terms` textually. **Agreement is recorded,
never treated as truth** — different AIs share training blind spots.

## Configuration (all `SKYNET_`-prefixed, see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `SKYNET_ENABLE_EXTERNAL_COMMS` | `false` | Feature flag (dark by default) |
| `SKYNET_AI_PROVIDERS` | `mock` | Activated providers; unknown names fail loudly |
| `SKYNET_AI_CONVERSATION_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `SKYNET_AI_MAX_TURNS_PER_CONVERSATION` | `5` | Hard turn budget |
| `SKYNET_AI_MAX_RETRIES` | `1` | Retries for transient failures |
| `SKYNET_AI_MAX_REQUEST_COST_USD` | unset | Optional cost ceiling |
| `SKYNET_AI_REQUEST_INTERVAL_SECONDS` | `1.0` | Politeness delay per provider |

### Conversation status semantics (stable contract)

- `completed` — the conversation closed cleanly, **including** conversations
  that used their full planned turn budget. Using all planned turns is normal
  operation, not a violation.
- `limit_reached` — a send was **rejected** because the budget was already
  exhausted: the caller wanted more turns than were budgeted. Recorded with an
  error note; valuable signal for the future learning loop (a planning hint
  that budgets were mis-sized).
- `failed` — provider failure or timeout; the conversation closed with the
  error preserved.

## CLI

```bash
# Offline, deterministic demonstration (mock provider):
.venv/Scripts/python.exe -m agency.cli ai-ask mock:mock-agent-1 \
    "What are the main approaches to long-term memory in AI agents?" \
    --storage memory

# Comparison across three participants (fully offline):
.venv/Scripts/python.exe -m agency.cli ai-demo \
    "How should autonomous agents persist knowledge?" --storage memory
```

Exit codes: `0` run completed and evaluation passed, `1` otherwise,
`--json` for machine-readable output.

## Trace events

All events flow through the run's standard `Trace` facade (one event stream,
one `run_id`, one ordering): `AI_PARTICIPANT_DISCOVERED`,
`AI_CONVERSATION_STARTED`, `AI_MESSAGE_SENT`, `AI_RESPONSE_RECEIVED`,
`AI_CONVERSATION_CONTINUED`, `AI_RESPONSE_ANALYZED`,
`AI_CONVERSATION_COMPLETED`, `AI_CONVERSATION_FAILED`.

## Adding a provider

1. Subclass `AIProvider` (`agency/comms/providers.py`) — implement
   `identify()`, `capabilities()`, and `send(request) -> ProviderResponse`.
   Read credentials from the environment; never hard-code or log them.
2. Export it and add an entry to `build_provider()`.
3. Name it in `SKYNET_AI_PROVIDERS` (plus its key in `.env`).
4. Add tests using the same scenarios the mock supports
   (`normal`, `multi_turn`, `timeout`, `failure`, `conflict`, `malicious`).

Real providers (anthropic/openai) land with the intelligence phase; the
interface, limits, provenance, and security boundary are already in place.
