"""Interactive terminal client for the SupportPilot API.

    uv run python scripts/chat.py --user cust_001

Streams replies token by token. When the agent asks for confirmation you can
approve, reject, or skip (leave it pending, e.g. to restart the server and
confirm later with `/confirm <thread_id> approve`).
"""

import argparse
import json
import uuid

import httpx


def stream_chat(client: httpx.Client, user: str, session: str, message: str) -> dict | None:
    confirmation = None
    with client.stream(
        "POST", "/chat", json={"message": message, "session_id": session}, headers={"X-User-Id": user}
    ) as resp:
        if resp.status_code != 200:
            resp.read()
            print(f"[HTTP {resp.status_code}] {resp.text}")
            return None
        event = None
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip() or "{}")
                if event == "token":
                    print(data["text"], end="", flush=True)
                elif event == "route":
                    print(f"[route: {data['intent']}] ", end="", flush=True)
                elif event == "confirmation_required":
                    confirmation = data
                elif event == "error":
                    print(f"\n[error] {data['message']}")
    print()
    return confirmation


def send_confirm(client: httpx.Client, user: str, thread_id: str, decision: str) -> None:
    resp = client.post(
        "/confirm", json={"thread_id": thread_id, "decision": decision}, headers={"X-User-Id": user}
    )
    if resp.status_code != 200:
        print(f"[HTTP {resp.status_code}] {resp.json().get('detail')}")
        return
    body = resp.json()
    print(body.get("reply") or json.dumps(body, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="cust_001")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--session", default=None)
    args = parser.parse_args()
    session = args.session or uuid.uuid4().hex[:12]
    print(f"user={args.user} session={session}  (commands: /confirm <thread_id> approve|reject, /quit)")

    with httpx.Client(base_url=args.url, timeout=120) as client:
        while True:
            try:
                message = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                continue
            if message == "/quit":
                break
            if message.startswith("/confirm"):
                parts = message.split()
                if len(parts) != 3:
                    print("usage: /confirm <thread_id> approve|reject")
                    continue
                send_confirm(client, args.user, parts[1], parts[2])
                continue

            print("bot> ", end="", flush=True)
            confirmation = stream_chat(client, args.user, session, message)
            if confirmation:
                print("\n[confirmation required]")
                print(f"  order:   {confirmation['order_id']}")
                print(f"  current: {confirmation['current_address']}")
                print(f"  new:     {confirmation['new_address']}")
                print(f"  thread:  {confirmation['thread_id']}")
                choice = input("approve / reject / skip? ").strip().lower()
                if choice in ("approve", "reject"):
                    print("bot> ", end="")
                    send_confirm(client, args.user, confirmation["thread_id"], choice)
                else:
                    print(f"Left pending. Later: /confirm {confirmation['thread_id']} approve")


if __name__ == "__main__":
    main()
