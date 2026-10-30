"""Playback agent for a Quest connected to ANOTHER machine (e.g. a laptop on the same Tailscale
network as the generation server).

It keeps a websocket open to the server (/ws/player); while connected, the server sends every
"play"/"stop" here instead of to a Quest on the server machine. The agent downloads the wav from
the server and plays it on the local Quest via OpenXR (Quest Link / Air Link).

  python scripts/remote_player.py --server http://desktop-nst9a4u:7860 [--name laptop]

With --audio-device the vibration goes to that sound output instead of the Quest (e.g. a speaker or an
amplifier driving a vibrotactile actuator), so narration can stay on the default output (e.g. AirPods):

  python scripts/remote_player.py --list-audio
  python scripts/remote_player.py --server http://desktop-nst9a4u:7860 --audio-device "Speakers"

Needs only: numpy, soundfile, pyopenxr, websockets, sounddevice (requirements-player.txt).
"""
import argparse
import io
import json
import socket
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf
from websockets.sync.client import connect


class AudioDevicePlayer:
    """Plays the vibration waveform on one sound output device; same interface as QuestHapticPlayer."""

    def __init__(self, device):
        import sounddevice as sd
        self.sd = sd
        self.device = find_device(device)
        info = sd.query_devices(self.device)
        self.rate = int(info["default_samplerate"])
        self.channels = min(2, int(info["max_output_channels"]))
        self.status = f"준비됨 (오디오 출력: {info['name']})"
        self.connected, self._end = True, 0.0

    def start(self):
        return self

    @property
    def is_playing(self):
        return time.time() < self._end

    def play(self, wav, sr, hand="both", gain=1.0, loop=1):
        w = np.clip(np.tile(np.asarray(wav, dtype=np.float32), max(1, int(loop))) * gain, -1, 1)
        if sr != self.rate:   # linear resampling is enough for vibration-band content
            w = np.interp(np.arange(int(len(w) * self.rate / sr)) * sr / self.rate, np.arange(len(w)), w).astype(np.float32)
        out = np.repeat(w[:, None], self.channels, axis=1)
        if self.channels == 2 and hand in ("left", "right"):
            out[:, 1 if hand == "left" else 0] = 0
        self.sd.play(out, self.rate, device=self.device)
        self._end = time.time() + len(w) / self.rate

    def stop(self):
        self.sd.stop()
        self._end = 0.0

    def close(self):
        self.stop()


def find_device(name):
    """Output device by index or (case-insensitive) name substring."""
    import sounddevice as sd
    if str(name).isdigit():
        return int(name)
    outs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
    hits = [i for i, d in outs if str(name).lower() in d["name"].lower()]
    if not hits:
        raise SystemExit(f"output device '{name}' not found; run with --list-audio")
    return hits[0]


def list_audio():
    import sounddevice as sd
    default_out = sd.default.device[1]
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f"{'*' if i == default_out else ' '} {i:3d}  {d['name']}  [{api}, {d['max_output_channels']}ch]")
    print("(* = Windows default output; narration in the browser goes there)")


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
    ap.add_argument("--server", help="e.g. http://desktop-nst9a4u:7860")
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--audio-device", help="play vibration on this sound output (index or name part) instead of the Quest")
    ap.add_argument("--list-audio", action="store_true", help="list sound outputs and exit")
    a = ap.parse_args()
    if a.list_audio:
        list_audio()
        return
    if not a.server:
        ap.error("--server is required")
    base = ipv4_base(a.server.rstrip("/"))
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/player?name={a.name}"
    if a.audio_device:
        player = AudioDevicePlayer(a.audio_device).start()
    else:
        from quest_haptic_play import QuestHapticPlayer
        player = QuestHapticPlayer().start()
    print(f"[player] {player.status}")

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
