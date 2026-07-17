"""Durable escalation/adjudication state-machine coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _escalated_task(conn) -> tuple[str, str]:
    task_id = kb.create_task(conn, title="BUI-53", assignee="builder")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="builder") is not None
    reason = "company.e2e-spec.ts expected 200 but received 404"
    assert kb.block_task(conn, task_id, kind="transient", reason=reason)
    assert kb.unblock_task(conn, task_id)
    assert kb.claim_task(conn, task_id, claimer="builder") is not None
    assert kb.block_task(conn, task_id, kind="transient", reason=reason)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "escalated"
    assert task.block_fingerprint
    return task_id, task.block_fingerprint


def test_fingerprint_normalizes_volatile_runtime_values() -> None:
    first = kb.compute_block_fingerprint(
        "transient",
        "container abcdef1234567890 at /tmp/run-1234/spec failed 2026-07-16T21:30:00Z",
        command="pnpm nx e2e api",
        target="company.e2e-spec.ts",
        phase="verification",
        error_class="AssertionError",
        assertion="expected 200 received 404",
    )
    second = kb.compute_block_fingerprint(
        "transient",
        "container fedcba0987654321 at /var/tmp/run-9999/spec failed 2026-07-17T01:15:00Z",
        command="pnpm nx e2e api",
        target="company.e2e-spec.ts",
        phase="verification",
        error_class="AssertionError",
        assertion="expected 200 received 404",
    )
    reaper = kb.compute_block_fingerprint(
        "transient",
        "Reaper failed",
        command="pnpm nx e2e api",
        target="company.e2e-spec.ts",
        phase="preflight",
        error_class="ContainerLaunchException",
        assertion="Ryuk unavailable",
    )
    assert first == second
    assert reaper != first


def test_adjudication_lifecycle_is_atomic_and_idempotent(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, fingerprint = _escalated_task(conn)
        decision = kb.create_adjudication(
            conn,
            task_id,
            decision_id=f"adjudicate:{task_id}:{fingerprint}:9",
            fingerprint=fingerprint,
            trigger_event_id=9,
            evidence=["run:2"],
        )
        duplicate = kb.create_adjudication(
            conn,
            task_id,
            decision_id=decision.decision_id,
            fingerprint=fingerprint,
            trigger_event_id=9,
        )
        assert duplicate.decision_id == decision.decision_id
        assert len(kb.list_adjudications(conn, task_id)) == 1

        started = kb.start_adjudication(
            conn,
            decision.decision_id,
            worker_session_id="session-adjudicator-1",
            child_task_id="child-1",
        )
        started_again = kb.start_adjudication(
            conn,
            decision.decision_id,
            worker_session_id="session-adjudicator-1",
            child_task_id="child-1",
        )
        assert started.status == "running"
        assert started_again.attempts == 1

        decided = kb.decide_adjudication(
            conn,
            decision.decision_id,
            verdict="resume",
            evidence=["head:pass-twice", "base:not-reproduced"],
            action="return to builder with a distinct diagnosis",
            resume_profile="builder",
        )
        assert decided.status == "decided"
        applied = kb.apply_adjudication(conn, decision.decision_id)
        applied_again = kb.apply_adjudication(conn, decision.decision_id)
        assert applied.status == "applied"
        assert applied_again.applied_at == applied.applied_at
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"


def test_escalated_task_cannot_use_generic_unblock(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, _ = _escalated_task(conn)
        assert kb.unblock_task(conn, task_id) is False
        assert kb.get_task(conn, task_id).status == "escalated"


def test_needs_human_keeps_parent_escalated(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, fingerprint = _escalated_task(conn)
        decision = kb.create_adjudication(
            conn,
            task_id,
            decision_id="decision-human",
            fingerprint=fingerprint,
            action="Choose whether the API contract should return 404 or 200",
        )
        kb.start_adjudication(
            conn, decision.decision_id, worker_session_id="session-human"
        )
        kb.decide_adjudication(
            conn,
            decision.decision_id,
            verdict="needs_human",
            evidence=["base reproduces inconsistently"],
        )
        applied = kb.apply_adjudication(conn, decision.decision_id)
        assert applied.status == "needs_human"
        assert kb.get_task(conn, task_id).status == "escalated"


def test_worker_session_is_required_and_cannot_be_stolen(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, fingerprint = _escalated_task(conn)
        decision = kb.create_adjudication(
            conn,
            task_id,
            decision_id="decision-session",
            fingerprint=fingerprint,
        )
        with pytest.raises(ValueError, match="worker_session_id"):
            kb.start_adjudication(
                conn, decision.decision_id, worker_session_id=" "
            )
        kb.start_adjudication(
            conn, decision.decision_id, worker_session_id="session-a"
        )
        with pytest.raises(ValueError, match="another worker session"):
            kb.start_adjudication(
                conn, decision.decision_id, worker_session_id="session-b"
            )


def test_verdict_requires_running_worker_session(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id, fingerprint = _escalated_task(conn)
        decision = kb.create_adjudication(
            conn,
            task_id,
            decision_id="decision-no-session",
            fingerprint=fingerprint,
        )
        with pytest.raises(ValueError, match="worker_session_id"):
            kb.decide_adjudication(
                conn, decision.decision_id, verdict="needs_human"
            )


def test_legacy_migration_only_moves_loop_breaker_triage(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        idea_id = kb.create_task(conn, title="rough idea", triage=True)
        loop_id = kb.create_task(conn, title="legacy loop", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='transient', block_recurrences=2 "
                "WHERE id=?",
                (loop_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'block_loop_detected', ?, 10)",
                (
                    loop_id,
                    '{"reason":"company.e2e-spec.ts expected 200 received 404"}',
                ),
            )

    kb.init_db()
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, idea_id).status == "triage"
        migrated = kb.get_task(conn, loop_id)
        assert migrated is not None
        assert migrated.status == "escalated"
        assert migrated.block_fingerprint

    # Idempotent re-open must not append another migration event.
    kb.init_db()
    with kb.connect_closing() as conn:
        events = [
            event for event in kb.list_events(conn, loop_id)
            if event.kind == "legacy_block_loop_migrated"
        ]
        assert len(events) == 1


def test_legacy_migration_ignores_cleared_loop(kanban_home: Path) -> None:
    """A stale loop event must not turn a later genuine triage idea escalated."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="reopened idea", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='transient', block_recurrences=2 "
                "WHERE id=?",
                (task_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'block_loop_detected', ?, 10)",
                (task_id, '{"reason":"old failure"}'),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'status', ?, 11)",
                (task_id, '{"status":"triage"}'),
            )

    kb.init_db()
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "triage"


def test_legacy_block_events_receive_independent_fingerprints(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="BUI-53 legacy", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='transient', block_recurrences=2 "
                "WHERE id=?",
                (task_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'blocked', ?, 10)",
                (task_id, '{"kind":"transient","reason":"expected 200 received 404"}'),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'block_loop_detected', ?, 11)",
                (task_id, '{"kind":"transient","reason":"Ryuk Reaper failed to start"}'),
            )

    kb.init_db()
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "escalated"
        events = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind in {"blocked", "block_loop_detected"}
        ]
        assert len(events) == 2
        assert events[0].payload["fingerprint"] != events[1].payload["fingerprint"]
        assert events[0].payload["recurrences"] == 1
        assert events[1].payload["recurrences"] == 1
        assert task.block_fingerprint == events[1].payload["fingerprint"]
        assert task.block_recurrences == 1


def test_legacy_recurrence_replay_resets_after_completion(kanban_home: Path) -> None:
    """Historical failures before a successful run do not consume its budget."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="BUI-53 completed then retried", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='transient', block_recurrences=1 "
                "WHERE id=?",
                (task_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'blocked', ?, 10)",
                (task_id, '{"kind":"transient","reason":"404"}'),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'completed', NULL, 11)",
                (task_id,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'blocked', ?, 12)",
                (task_id, '{"kind":"transient","reason":"404"}'),
            )

    kb.init_db()
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.block_recurrences == 1
