"""Tests for the cron inactivity watchdog: the daemon-thread loop and the timeout it raises."""

import threading
import time

import pytest


class TestInactivityWatchdogLoop:
    """The daemon-thread inactivity helper must not depend on the caller thread."""

    def test_fires_when_idle_crosses_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 0.0}
        results: list = []

        def _watch():
            results.append(
                _inactivity_watchdog_loop(
                    get_idle_seconds=lambda: idle["s"],
                    limit_s=0.2,
                    poll_s=0.05,
                    stop=stop,
                    future_done=lambda: False,
                )
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        time.sleep(0.12)
        idle["s"] = 1.0
        watcher.join(timeout=2.0)
        stop.set()
        assert results == [True]
        assert not watcher.is_alive()

    def test_stops_when_future_completes_before_idle_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        fired = _inactivity_watchdog_loop(
            get_idle_seconds=lambda: 0.0,
            limit_s=10.0,
            poll_s=0.05,
            stop=stop,
            future_done=lambda: True,
        )
        assert fired is False

    def test_fires_while_caller_thread_is_blocked(self):
        """#94285: a blocked run_job thread must not disable the watchdog."""
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 1.0}
        result = {"fired": None}

        def _watch():
            result["fired"] = _inactivity_watchdog_loop(
                get_idle_seconds=lambda: idle["s"],
                limit_s=0.15,
                poll_s=0.05,
                stop=stop,
                future_done=lambda: False,
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        # Simulate the family-A stall: this thread cannot poll.
        time.sleep(0.4)
        watcher.join(timeout=2.0)
        stop.set()
        assert result["fired"] is True
        assert not watcher.is_alive()


class _SequenceAgent:
    """Agent whose activity sample moves on between reads (a host resumed, a tool unblocked)."""

    def __init__(self, *summaries):
        self._summaries = [dict(s) for s in summaries]
        self.reads = 0
        self.latched = threading.Event()

    def run_conversation(self, prompt, task_id=None, **_kwargs):
        # Stay alive until the watchdog has latched — and the caller has recorded that latch —
        # so the run ends through the inactivity path, not the future-completed path.
        assert self.latched.wait(5.0), "watchdog stub never latched"
        time.sleep(0.1)
        return {"final_response": "unused", "messages": []}

    def get_activity_summary(self):
        index = min(self.reads, len(self._summaries) - 1)
        self.reads += 1
        return dict(self._summaries[index])


_STALLED_SAMPLE = {
    "last_activity_desc": "waiting on provider stream",
    "seconds_since_activity": 612.0,
    "current_tool": None,
    "api_call_count": 0,
    "max_iterations": 10,
}
_RESUMED_SAMPLE = {
    "last_activity_desc": "terminal command running",
    "seconds_since_activity": 3.0,
    "current_tool": "terminal",
    "api_call_count": 1,
    "max_iterations": 10,
}


class TestInactivityTimeoutReportsTheLatchedSample:
    """The raised timeout reports the sample the watchdog latched the limit on (#127775).

    The watchdog latches on the LAST sample it reads, so its caller has to keep that sample:
    re-reading ``get_activity_summary()`` at raise time reports whatever the agent is doing
    afterwards, which is how a run whose ledger row spans 3668s reported
    "idle for 3s (limit 600s)" as soon as the host resumed.
    """

    def test_timeout_message_uses_the_latched_sample(self, monkeypatch):
        """The watchdog's caller must hand the latched sample to the raise path."""
        import cron.scheduler as sched

        agent = _SequenceAgent(_STALLED_SAMPLE, _RESUMED_SAMPLE)
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")

        def _latches_immediately(*, get_idle_seconds, limit_s, poll_s, stop, future_done):
            # The real loop samples here and returns True on the sample that crossed the limit;
            # its own polling behaviour is covered by TestInactivityWatchdogLoop.
            get_idle_seconds()
            agent.latched.set()
            return True

        monkeypatch.setattr(sched, "_inactivity_watchdog_loop", _latches_immediately)

        with pytest.raises(TimeoutError) as excinfo:
            sched._run_agent_with_watchdog(
                agent, "prompt", {"id": "test-job", "name": "test-job"},
                "test-job", "test-job", "task-id", None,
            )

        message = str(excinfo.value)
        assert "idle for 612s (limit 600s)" in message
        assert "last activity: waiting on provider stream" in message
        assert "idle for 3s" not in message

    def test_without_a_latched_sample_the_live_read_is_still_reported(self):
        """A caller that holds no sample keeps the previous behaviour."""
        from cron.scheduler import _raise_inactivity_timeout

        agent = _SequenceAgent(_RESUMED_SAMPLE)

        with pytest.raises(TimeoutError) as excinfo:
            _raise_inactivity_timeout(agent, "job", 600.0)

        message = str(excinfo.value)
        assert "idle for 3s (limit 600s)" in message
        assert "last activity: terminal command running" in message

