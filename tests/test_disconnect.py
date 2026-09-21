"""Connection loss must complete each pending cursor exactly once."""
import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pandas
import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.protocol import State

from wherobots.db.connection import Connection
from wherobots.db.driver import connect_direct
from wherobots.db.errors import OperationalError


class Transport:
    def __init__(self):
        self.protocol = MagicMock(state=State.OPEN)
        self.incoming = queue.Queue()
        self.sent = []

    def recv(self, timeout):
        value = self.incoming.get(timeout=3)
        if isinstance(value, Exception):
            self.protocol.state = State.CLOSED
            raise value
        return json.dumps(value)

    def send(self, value):
        self.sent.append(json.loads(value))

    def close(self):
        self.incoming.put(ConnectionClosedOK(None, None))


@pytest.mark.parametrize(
    "error",
    [
        ConnectionClosedError(None, None),
        ConnectionClosedOK(None, None),
        OSError("transport lost"),
    ],
)
def test_disconnect_unblocks_all_cursors_and_rejects_new_queries(error):
    ws = Transport()
    conn = Connection(ws, session_id="session-1")
    cursors = [conn.cursor() for _ in range(3)]
    for cursor in cursors:
        cursor.execute("MERGE INTO secret VALUES ('private')")
    ws.incoming.put(error)
    conn._Connection__thread.join(timeout=3)
    assert not conn._Connection__thread.is_alive()
    for cursor, request in zip(cursors, ws.sent):
        with pytest.raises(OperationalError) as exc:
            cursor.fetchall()
        assert "session-1" in str(exc.value)
        assert request["execution_id"] in str(exc.value)
        assert "Commit outcome is unknown" in str(exc.value)
        assert "private" not in str(exc.value)
        with pytest.raises(OperationalError):
            cursor.fetchall()
    assert not conn._Connection__queries
    with pytest.raises(OperationalError):
        conn.cursor().execute("INSERT INTO t VALUES (1)")
    assert len(ws.sent) == 3


def test_delivered_result_wins_close_and_is_not_overwritten():
    ws = Transport()
    conn = Connection(ws)
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    with patch.object(
        conn, "_handle_results", return_value=pandas.DataFrame({"x": [1]})
    ):
        ws.incoming.put(
            {
                "kind": "execution_result",
                "execution_id": ws.sent[0]["execution_id"],
                "state": "succeeded",
                "results": {"ignored": True},
            }
        )
        ws.incoming.put(ConnectionClosedOK(None, None))
        conn._Connection__thread.join(timeout=3)
    assert cursor.fetchall()["x"].tolist() == [1]
    assert cursor._Cursor__queue.empty()


def test_close_fails_pending_without_waiting_for_status():
    ws = Transport()
    details = MagicMock()
    conn = Connection(ws, failure_details=details)
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    conn.close()
    with pytest.raises(OperationalError):
        cursor.fetchall()
    details.assert_not_called()
    conn._Connection__thread.join(timeout=3)


def test_stalled_enrichment_is_bounded_once_for_all_cursors():
    release = threading.Event()
    ws = Transport()

    def lookup():
        release.wait(timeout=10)
        return "late"

    conn = Connection(ws, failure_details=lookup)
    cursors = [conn.cursor() for _ in range(3)]
    for cursor in cursors:
        cursor.execute("SELECT 1")
    started = time.monotonic()
    ws.incoming.put(ConnectionClosedError(None, None))
    conn._Connection__thread.join(timeout=3)
    try:
        assert not conn._Connection__thread.is_alive()
        assert time.monotonic() - started < 3
        for cursor in cursors:
            with pytest.raises(OperationalError, match="Commit outcome is unknown"):
                cursor.fetchall()
    finally:
        release.set()


@pytest.mark.parametrize(
    "status,payload",
    [
        (200, {"firstFailure": {"message": "Evicted: ephemeral-storage"}}),
        (404, {}),
        (503, {}),
        (200, {}),
        (200, None),
    ],
)
def test_http_enrichment_best_effort(status, payload):
    ws = Transport()
    response = MagicMock(status_code=status)
    response.json.return_value = payload
    response.__enter__.return_value = response
    with patch(
        "wherobots.db.driver.websockets.sync.client.connect", return_value=ws
    ), patch("wherobots.db.driver.requests.get", return_value=response) as get:
        conn = connect_direct(
            "wss://compute/sql",
            headers={"Authorization": "Bearer test"},
            session_status_url="https://api/sql/session/session-1",
        )
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        ws.incoming.put(ConnectionClosedError(None, None))
        conn._Connection__thread.join(timeout=3)
        assert not conn._Connection__thread.is_alive()
        with pytest.raises(OperationalError) as exc:
            cursor.fetchall()
        assert ("ephemeral-storage" in str(exc.value)) == (
            status == 200 and bool(payload)
        )
        assert get.call_args.kwargs["timeout"] == 1.0
        assert get.call_args.kwargs["allow_redirects"] is False


def test_send_failure_does_not_leave_pending_query():
    ws = Transport()
    conn = Connection(ws)
    ws.send = MagicMock(side_effect=ConnectionClosedError(None, None))
    cursor = conn.cursor()
    cursor.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(OperationalError):
        cursor.fetchall()
    assert not conn._Connection__queries
    conn.close()
    conn._Connection__thread.join(timeout=3)


def test_buffered_result_is_drained_even_when_transport_is_already_closed():
    ws = Transport()
    with patch("wherobots.db.connection.threading.Thread.start"):
        conn = Connection(ws)
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    ws.incoming.put(
        {
            "kind": "execution_result",
            "execution_id": ws.sent[0]["execution_id"],
            "state": "succeeded",
            "results": {"ignored": True},
        }
    )
    ws.incoming.put(ConnectionClosedOK(None, None))
    ws.protocol.state = State.CLOSED
    with patch.object(
        conn, "_handle_results", return_value=pandas.DataFrame({"x": [1]})
    ):
        conn._Connection__main_loop()
    assert cursor.fetchall()["x"].tolist() == [1]


@pytest.mark.parametrize(
    "error", [OSError("HTTP unavailable"), ValueError("invalid JSON")]
)
def test_enrichment_errors_preserve_connection_failure(error):
    ws = Transport()
    conn = Connection(ws, failure_details=MagicMock(side_effect=error))
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    ws.incoming.put(ConnectionClosedError(None, None))
    conn._Connection__thread.join(timeout=3)
    assert not conn._Connection__thread.is_alive()
    with pytest.raises(OperationalError, match="Commit outcome is unknown"):
        cursor.fetchall()


def test_enrichment_error_is_logged_at_debug(caplog):
    ws = Transport()
    conn = Connection(
        ws, failure_details=MagicMock(side_effect=ValueError("invalid JSON"))
    )
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    with caplog.at_level("DEBUG"):
        ws.incoming.put(ConnectionClosedError(None, None))
        conn._Connection__thread.join(timeout=3)
    assert "Failure-details lookup failed: invalid JSON" in caplog.text


def test_enrichment_thread_start_error_is_logged_at_debug(caplog):
    ws = Transport()
    conn = Connection(ws, failure_details=MagicMock(return_value="details"))
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    with caplog.at_level("DEBUG"), patch(
        "wherobots.db.connection.threading.Thread.start",
        side_effect=RuntimeError("thread unavailable"),
    ):
        ws.incoming.put(ConnectionClosedError(None, None))
        conn._Connection__thread.join(timeout=3)
    assert "Could not start failure-details lookup: thread unavailable" in caplog.text


@pytest.mark.parametrize(
    "decode_error", [ValueError("malformed payload"), OSError("decoder I/O error")]
)
def test_result_decode_error_does_not_fail_other_queries(decode_error):
    decoded = threading.Event()
    ws = Transport()
    conn = Connection(ws)
    bad_cursor = conn.cursor()
    good_cursor = conn.cursor()
    bad_cursor.execute("SELECT bad")
    good_cursor.execute("SELECT good")

    def decode(execution_id, results):
        if execution_id == ws.sent[0]["execution_id"]:
            decoded.set()
            raise decode_error
        return pandas.DataFrame({"x": [1]})

    with patch.object(conn, "_handle_results", side_effect=decode):
        for request in ws.sent:
            ws.incoming.put(
                {
                    "kind": "execution_result",
                    "execution_id": request["execution_id"],
                    "state": "succeeded",
                    "results": {"ignored": True},
                }
            )
        assert decoded.wait(timeout=3)
        assert good_cursor.fetchall()["x"].tolist() == [1]
    assert conn._Connection__thread.is_alive()
    conn.close()
    with pytest.raises(OperationalError):
        bad_cursor.fetchall()
    conn._Connection__thread.join(timeout=3)


def test_close_waits_for_registered_query_to_be_sent():
    send_started = threading.Event()
    release_send = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    events = []
    ws = Transport()

    def send(value):
        send_started.set()
        assert release_send.wait(timeout=3)
        ws.sent.append(json.loads(value))
        events.append("send")

    def close():
        events.append("close")
        ws.incoming.put(ConnectionClosedOK(None, None))

    ws.send = send
    ws.close = close
    conn = Connection(ws)
    cursor = conn.cursor()
    execute_thread = threading.Thread(target=cursor.execute, args=("SELECT 1",))
    execute_thread.start()
    assert send_started.wait(timeout=3)

    def close_connection():
        close_started.set()
        conn.close()
        close_finished.set()

    close_thread = threading.Thread(target=close_connection)
    close_thread.start()
    assert close_started.wait(timeout=3)
    assert not close_finished.is_set()
    release_send.set()
    execute_thread.join(timeout=3)
    close_thread.join(timeout=3)
    conn._Connection__thread.join(timeout=3)
    assert not execute_thread.is_alive()
    assert not close_thread.is_alive()
    assert events == ["send", "close"]
    with pytest.raises(OperationalError):
        cursor.fetchall()


def test_close_racing_result_decode_delivers_only_one_terminal_outcome():
    decoding = threading.Event()
    release = threading.Event()
    ws = Transport()
    conn = Connection(ws)
    cursor = conn.cursor()
    cursor.execute("SELECT 1")

    def decode(*args):
        decoding.set()
        assert release.wait(timeout=3)
        return pandas.DataFrame({"x": [1]})

    with patch.object(conn, "_handle_results", side_effect=decode):
        ws.incoming.put(
            {
                "kind": "execution_result",
                "execution_id": ws.sent[0]["execution_id"],
                "state": "succeeded",
                "results": {"ignored": True},
            }
        )
        assert decoding.wait(timeout=3)
        conn.close()
        release.set()
        conn._Connection__thread.join(timeout=3)
    assert not conn._Connection__thread.is_alive()
    with pytest.raises(OperationalError):
        cursor.fetchall()
    assert cursor._Cursor__queue.empty()
