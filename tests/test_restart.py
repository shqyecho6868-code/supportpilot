"""Verification 3: a turn paused for confirmation survives a server restart.

Runs the real app under uvicorn in a subprocess, kills it with SIGKILL while the
turn is waiting for confirmation, starts a brand-new process, and confirms.
The only thing connecting the two processes is the Postgres checkpoint.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from tests.conftest import ALICE, TEST_DATABASE_URL, TEST_REDIS_URL, fetch_order, parse_sse

ROOT = Path(__file__).resolve().parents[1]
CHANGE = "Please change the shipping address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, port: int, llm_provider: str = "rule_based"):
        self.port = port
        self.llm_provider = llm_provider
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = {
            **os.environ,
            "DATABASE_URL": TEST_DATABASE_URL,
            "REDIS_URL": TEST_REDIS_URL,
            "LLM_PROVIDER": self.llm_provider,
            "LANGFUSE_ENABLED": "false",
            "LONG_TERM_MEMORY": "false",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "supportpilot.api.app:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        self.kill()
        raise RuntimeError("server did not become healthy")

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGKILL)  # a crash, not a graceful shutdown
            self.proc.wait(timeout=10)


async def _post(server: Server, path: str, **kwargs) -> httpx.Response:
    # A fresh client per call: pooled connections would point at a killed process.
    async with httpx.AsyncClient(base_url=server.url) as client:
        return await client.post(path, **kwargs)


async def _get(server: Server, path: str) -> httpx.Response:
    async with httpx.AsyncClient(base_url=server.url, timeout=1) as client:
        return await client.get(path)


@pytest.fixture
def server(clean_state) -> Iterator[Server]:
    srv = Server(_free_port())
    yield srv
    srv.kill()


async def test_pending_confirmation_survives_restart(server):
    before = await fetch_order("ORD-1001")

    server.start()
    first_pid = server.proc.pid
    resp = await _post(server, "/chat", json={"message": CHANGE}, headers={"X-User-Id": ALICE}, timeout=30)
    events = dict(parse_sse(resp.text))
    thread_id = events["confirmation_required"]["thread_id"]

    server.kill()
    with pytest.raises(httpx.ConnectError):
        await _get(server, "/health")
    assert await fetch_order("ORD-1001") == before

    server.start()
    assert server.proc.pid != first_pid
    resp = await _post(
        server,
        "/confirm",
        json={"thread_id": thread_id, "decision": "approve"},
        headers={"X-User-Id": ALICE},
        timeout=30,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"
    row = await fetch_order("ORD-1001")
    assert row["address"]["city"] == "San Francisco"
    assert row["address_events"] == 1

    # And after yet another restart, the finished turn cannot be confirmed again.
    server.kill()
    server.start()
    resp = await _post(
        server,
        "/confirm",
        json={"thread_id": thread_id, "decision": "approve"},
        headers={"X-User-Id": ALICE},
        timeout=30,
    )
    assert resp.status_code == 409
    assert (await fetch_order("ORD-1001"))["address_events"] == 1
