"""Save 1280x720 screenshots of the dashboard pages with headless Edge (or Chrome).

    python scripts/screenshot_pages.py <out_dir> [base_url] [pages] [wait_seconds]

Defaults: base_url http://localhost:8501, pages "worklist,card,insights,roi", wait 10 s per page.
Drives the browser through the DevTools protocol because the plain `--screenshot` flag captures
Streamlit's loading skeleton (it does not wait for the websocket). Needs the `websockets` package,
which uvicorn already installs in the project venv.
"""
from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

import websockets

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
PORT = 9333


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    base = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8501").rstrip("/")
    pages = (sys.argv[3] if len(sys.argv) > 3 else "worklist,card,insights,roi").split(",")
    wait = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    out.mkdir(parents=True, exist_ok=True)
    browser = next((b for b in BROWSERS if pathlib.Path(b).exists()), None)
    if browser is None:
        raise SystemExit("no Edge or Chrome found")
    profile = out / ".browser-profile"
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
                             "--window-size=1280,720", f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
                             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        targets = None
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:
                time.sleep(0.5)
        if not targets:
            raise SystemExit("browser did not expose the DevTools port")
        ws_url = next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"]
        asyncio.run(shoot(ws_url, base, pages, wait, out))
    finally:
        proc.kill()


async def shoot(ws_url: str, base: str, pages: list[str], wait: float, out: pathlib.Path) -> None:
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        mid = 0

        async def call(method: str, **params):
            nonlocal mid
            mid += 1
            await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == mid:
                    return msg.get("result", msg)

        await call("Emulation.setDeviceMetricsOverride", width=1280, height=720, deviceScaleFactor=1, mobile=False)
        for page in pages:
            # the default page lives at the root; "worklist" is accepted as its name
            path = "" if page == "worklist" else page
            await call("Page.navigate", url=f"{base}/{path}")
            await asyncio.sleep(wait)
            shot = await call("Page.captureScreenshot", format="png")
            target = out / f"{page}.png"
            target.write_bytes(base64.b64decode(shot["data"]))
            print(f"saved {target}")


if __name__ == "__main__":
    main()
