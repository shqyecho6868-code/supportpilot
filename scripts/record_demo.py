"""Record the README demo: a real session with scripts/chat.py, rendered as a terminal GIF.

    docker compose up -d --build --wait          # the API on :8000
    uv run --with pillow python scripts/record_demo.py --out docs/demo.gif
    uv run python scripts/reset_db.py            # the demo changes ORD-1001's address

The client runs in a pseudo-terminal; the script types the scripted inputs at a human pace and
records every byte the client prints, with timestamps. The recording is then rendered frame by
frame (long waits for the model are shortened, nothing else is edited). Calls real Claude through
the API: about 6 Haiku calls, under $0.01.
"""

import argparse
import os
import pty
import re
import select
import sys
import time
from pathlib import Path

SCRIPT = [
    ("you> ", "Where is my order ORD-1002?"),
    ("you> ", "Please change the shipping address of ORD-1001 to 500 Market St, San Francisco, CA 94105, US"),
    ("approve / reject / skip? ", "approve"),
    ("you> ", "What's the shipping address on ORD-1001 now?"),
    ("you> ", "/quit"),
]
TYPING_DELAY = 0.035
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def record(user: str, url: str, timeout: float = 180.0) -> list[tuple[float, str]]:
    """Run the client in a pty, answer each prompt from SCRIPT; return (seconds, text) chunks."""
    root = Path(__file__).resolve().parents[1]
    pid, fd = pty.fork()
    if pid == 0:  # child: the real interactive client
        os.chdir(root)
        os.execvp("uv", ["uv", "run", "python", "scripts/chat.py", "--user", user, "--url", url])
    events: list[tuple[float, str]] = []
    started = time.monotonic()
    screen = ""
    steps = list(SCRIPT)
    typing: list[str] = []  # characters still to type, one per TYPING_DELAY
    next_key = 0.0
    deadline = started + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.01)
        if ready:
            try:
                data = os.read(fd, 4096).decode("utf-8", "replace")
            except OSError:
                break
            if not data:
                break
            events.append((time.monotonic() - started, data))
            screen += data
        now = time.monotonic()
        if typing and now >= next_key:
            os.write(fd, typing.pop(0).encode())
            next_key = now + TYPING_DELAY
        elif not typing and steps and screen.endswith(steps[0][0]):
            _, text = steps.pop(0)
            typing = list(text + "\r")
            next_key = now + 0.8  # a beat before typing
            screen = ""
        if not steps and not typing and not ready:
            try:
                if os.waitpid(pid, os.WNOHANG)[0]:
                    break
            except ChildProcessError:
                break
    return events


def _apply(lines: list[str], text: str, cols: int) -> None:
    """Minimal terminal: printable text, \\n, \\r, backspace; long lines wrap."""
    for ch in _ANSI.sub("", text):
        if ch == "\n":
            lines.append("")
        elif ch == "\r":
            continue  # the pty sends \r\n; a bare \r never occurs in this client's output
        elif ch == "\b":
            lines[-1] = lines[-1][:-1]
        elif ch.isprintable():
            if len(lines[-1]) >= cols:
                lines.append("")
            lines[-1] += ch


def render(events, out: Path, cols: int, rows: int, max_gap: float, step: float) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    cw = round(font.getlength("M"))
    ch = 20
    pad = 16
    bar = 30
    width, height = cols * cw + 2 * pad, rows * ch + 2 * pad + bar
    bg, fg, dim = (24, 26, 33), (220, 223, 228), (130, 137, 151)
    accent, green = (229, 192, 123), (152, 195, 121)
    title = "supportpilot: scripts/chat.py --user cust_001"

    # Compress the timeline: waits longer than max_gap are shortened to max_gap.
    timeline, t_prev, shift = [], 0.0, 0.0
    for t, text in events:
        gap = t - t_prev
        if gap > max_gap:
            shift += gap - max_gap
        timeline.append((t - shift, text))
        t_prev = t

    def color(line: str):
        if line.startswith("you>"):
            return accent
        if line.startswith(("[confirmation", "  order:", "  current:", "  new:", "  expires:", "  thread:")):
            return green
        if line.startswith(("user=", "[tokens")):
            return dim
        return fg

    def frame(lines: list[str]) -> Image.Image:
        img = Image.new("RGB", (width, height), bg)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, bar], fill=(40, 43, 52))
        for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([14 + i * 22, 9, 26 + i * 22, 21], fill=c)
        d.text((width // 2, bar // 2), title, fill=dim, font=font, anchor="mm")
        for i, line in enumerate(lines[-rows:]):
            d.text((pad, bar + pad + i * ch), line, fill=color(line), font=font)
        return img.quantize(colors=16, method=Image.Quantize.MEDIANCUT)

    frames, durations, lines = [], [], [""]
    last: list[str] = []
    end = timeline[-1][0] if timeline else 0.0
    i, t = 0, 0.0
    while t <= end + step:
        while i < len(timeline) and timeline[i][0] <= t:
            _apply(lines, timeline[i][1], cols)
            i += 1
        snapshot = list(lines)
        if frames and snapshot == last:
            durations[-1] += int(step * 1000)
        else:
            frames.append(frame(snapshot))
            durations.append(int(step * 1000))
            last = snapshot
        t += step
    durations[-1] += 4000  # hold the last frame
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    seconds, size = sum(durations) / 1000, out.stat().st_size / 1e6
    print(f"wrote {out}: {len(frames)} frames, {seconds:.1f} s, {size:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="cust_001")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="docs/demo.gif")
    parser.add_argument("--cols", type=int, default=112)
    parser.add_argument("--rows", type=int, default=26)
    parser.add_argument("--max-gap", type=float, default=1.2, help="longest pause kept, in seconds")
    args = parser.parse_args()
    events = record(args.user, args.url)
    if not events:
        sys.exit("nothing recorded: is the API running on --url?")
    raw = "".join(text for _, text in events)
    Path(args.out).with_suffix(".txt").write_text(_ANSI.sub("", raw).replace("\r", ""))
    render(events, Path(args.out), args.cols, args.rows, args.max_gap, step=0.08)


if __name__ == "__main__":
    main()
