"""Regression tests for memory provider selection during AIAgent init."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names




def test_aiagent_reuses_handed_in_memory_manager_without_reinitializing():
    """A caller that rebuilds the agent per turn (gateway api_server) hands back the session's manager:
    the provider keeps its state — no second load, no second initialize (#120116)."""
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}
    common = dict(
        api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
        skip_context_files=True, skip_memory=False, session_id="sess-api", platform="api_server",
    )
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider) as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        first = AIAgent(**common)
        manager = first._memory_manager
        assert manager is not None and load_memory_provider.call_count == 1
        provider.init_session_id = None  # a re-initialize would set it again

        second = AIAgent(memory_manager=manager, **common)

    assert second._memory_manager is manager
    assert load_memory_provider.call_count == 1
    assert provider.init_session_id is None


def _init_agent_with_provider(provider, session_id):
    """Build a real AIAgent whose ``memory.provider`` resolves to *provider*.

    Mirrors the harness above: config and the provider loader are patched, so the
    provider object under test is the one ``_init_memory`` probes during init.
    """
    cfg = {"memory": {"provider": provider.name}, "agent": {}}
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id=session_id,
        )


class _ProbeProvider:
    """Minimal provider shape — only what ``_init_memory`` touches during init."""

    name = "probe-provider"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass


class _SystemExitProbeProvider(_ProbeProvider):
    """Availability probe that dies the way an unguarded lazy-install takeover does.

    ``tools.lazy_deps.install_specs`` -> ``stop_for_relaunch()`` raises ``SystemExit``
    (see #122326); the memory-provider init path is one of its callers.
    """

    name = "sys-exit-probe-provider"

    def is_available(self) -> bool:
        raise SystemExit(1)


class _SystemExitReasonProvider(_ProbeProvider):
    """Unavailable provider whose ``unavailable_reason()`` raises ``SystemExit``."""

    name = "sys-exit-reason-provider"

    def is_available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        raise SystemExit(1)


def test_systemexit_from_is_available_does_not_kill_agent_init(caplog):
    """#123042: ``SystemExit`` is not an ``Exception`` — the guard must contain it.

    Before the fix the raise escaped ``init_agent`` and killed the process (exit code 1
    with no output in the CLI, a crash loop in the gateway). A failed provider is a
    degraded feature; it must never end the process.
    """
    escaped = None
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        try:
            agent = _init_agent_with_provider(_SystemExitProbeProvider(), "sess-sysexit")
        except SystemExit as exc:  # the defect: a probe's SystemExit leaves init_agent
            escaped, agent = exc, None

    assert escaped is None, (
        f"SystemExit({escaped.code if escaped else ''}) escaped agent init — a memory "
        "provider's availability probe must not be able to kill the process"
    )
    assert agent is not None
    assert agent._memory_manager is None
    failed = [r for r in caplog.records if "Memory provider plugin init failed" in r.getMessage()]
    assert failed, [r.getMessage() for r in caplog.records]
    # The contained exception is reported, not swallowed: SystemExit(1) renders as "1".
    assert failed[0].getMessage().endswith(": 1")


def test_systemexit_from_unavailable_reason_does_not_kill_agent_init(caplog):
    """The nested probe is guarded too: ``suppress(Exception)`` misses ``SystemExit``."""
    escaped = None
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        try:
            agent = _init_agent_with_provider(_SystemExitReasonProvider(), "sess-sysexit-reason")
        except SystemExit as exc:
            escaped, agent = exc, None

    assert escaped is None, (
        f"SystemExit({escaped.code if escaped else ''}) escaped agent init — "
        "unavailable_reason() must not be able to kill the process either"
    )
    assert agent is not None
    assert agent._memory_manager is None
    unavailable = [r for r in caplog.records if "sys-exit-reason-provider" in r.getMessage()]
    assert unavailable, [r.getMessage() for r in caplog.records]


def test_keyboard_interrupt_from_is_available_still_propagates():
    """A user's Ctrl-C must keep working — only ``SystemExit`` is added to the guard.

    ``hermes_cli/plugins_loader.py`` states the contract for this class of guard:
    "SystemExit too ... KeyboardInterrupt still propagates".
    """

    class _InterruptProvider(RecordingMemoryProvider):
        name = "interrupt-provider"

        def is_available(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _init_agent_with_provider(_InterruptProvider(), "sess-interrupt")

