"""Playback agent for a Quest connected to ANOTHER machine (e.g. a laptop on the same Tailscale
network as the generation server).

It keeps a websocket open to the server (/ws/player); while connected, the server sends every
"play"/"stop" here instead of to a Quest on the server machine. The agent downloads the wav from
the server and plays it on the local Quest via OpenXR (Quest Link / Air Link).

  python scripts/remote_player.py --server http://desktop-nst9a4u:7860 [--name laptop]

Needs only: numpy, soundfile, pyopenxr, websockets (requirements-player.txt).
"""
import argparse
import io
import json
import socket
import threading
import time
import urllib.parse
import urllib.request

import soundfile as sf
from websockets.sync.client import connect

from quest_haptic_play import QuestHapticPlayer


def ipv4_base(url):
    """Resolve the server name to IPv4 once: the server listens on IPv4, and trying an IPv6 address
    first (MagicDNS / link-local) adds ~2 s to every download."""
    u = urllib.parse.urlsplit(url)
    try:
        ip = socket.getaddrinfo(u.hostname, u.port or 80, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except OSError:
        return url
    return urllib.parse.urlunsplit((u.scheme, f"{ip}:{u.port}" if u.port else ip, u.path, "", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="e.g. http://desktop-nst9a4u:7860")
    ap.add_argument("--name", default=socket.gethostname())
    a = ap.parse_args()
    base = ipv4_base(a.server.rstrip("/"))
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/player?name={a.name}"
    player = QuestHapticPlayer().start()
    print(f"[player] Quest: {player.status}")

    while True:
        try:
            with connect(ws_url, open_timeout=10) as ws:
                print(f"[player] connected to {base} as '{a.name}'")
                alive = threading.Event()
                alive.set()

                def report():
                    while alive.is_set():
                        try:
                            ws.send(json.dumps({"type": "status", "status": player.status, "connected": player.connected,
                                                "ready": player.status.startswith("준비됨") or player.status == "재생 중",
                                                "playing": player.is_playing}))
                        except Exception:  # noqa: BLE001  connection closed; the outer loop reconnects
                            return
                        time.sleep(1)

                threading.Thread(target=report, daemon=True).start()
                try:
                    for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("cmd") == "play":
                            t0 = time.time()
                            data = urllib.request.urlopen(base + msg["path"], timeout=30).read()
                            w, sr = sf.read(io.BytesIO(data), dtype="float32")
                            if w.ndim > 1:
                                w = w.mean(axis=1)
                            player.play(w, sr, msg.get("hand", "both"), msg.get("gain", 1.0), msg.get("loop", 1))
                            print(f"[player] play {len(w) / sr:.2f}s ({(time.time() - t0) * 1000:.0f} ms to start) -> {player.status}")
                        elif msg.get("cmd") == "stop":
                            player.stop()
                            print("[player] stop")
                finally:
                    alive.clear()
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001  server restarting / network down
            print(f"[player] disconnected ({type(exc).__name__}: {exc}); retrying in 3 s")
            time.sleep(3)
    player.close()


if __name__ == "__main__":
    main()
