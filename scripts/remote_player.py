"""Playback agent on the machine where the user sits (e.g. a laptop on the same Tailscale network as the
generation server).

It keeps a websocket open to the server (/ws/player) and reports its sound output devices; the web page
chooses, through the server, where things play:
  vibration -> the Quest controllers (OpenXR, Quest Link / Air Link) or any sound output
               (a speaker or an amplifier driving a vibrotactile actuator)
  voice     -> any sound output (narration of Story2Hap), started together with the vibration
The agent downloads the wavs from the server. The choice is remembered in ~/.subevent2hap_player.json.

  python scripts/remote_player.py --server http://desktop-nst9a4u:7860 [--name laptop]
  python scripts/remote_player.py --list-audio

Needs only: numpy, soundfile, websockets, sounddevice (+ pyopenxr for the Quest) (requirements-player.txt).
"""
import argparse
import io
import json
import pathlib
import socket
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf
from websockets.sync.client import connect

CONFIG = pathlib.Path.home() / ".subevent2hap_player.json"


def ipv4_base(url):
    """Resolve the server name to IPv4 once: the server listens on IPv4, and trying an IPv6 address
    first (MagicDNS / link-local) adds ~2 s to every download."""
    u = urllib.parse.urlsplit(url)
    try:
        ip = socket.getaddrinfo(u.hostname, u.port or 80, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except OSError:
        return url
    return urllib.parse.urlunsplit((u.scheme, f"{ip}:{u.port}" if u.port else ip, u.path, "", ""))


# ---------------------------------------------------------------- sound outputs
def output_devices():
    """Output devices of one host API. WASAPI first: its names come from wide strings, so non-English
    device names (e.g. Korean) are not garbled the way DirectSound/MME names can be on some systems."""
    import sounddevice as sd
    apis = [a["name"] for a in sd.query_hostapis()]
    api = next((i for pref in ("WASAPI", "DirectSound", "MME") for i, n in enumerate(apis) if pref in n), None)
    default_out = sd.default.device[1]
    default_name = sd.query_devices(default_out)["name"] if default_out is not None and default_out >= 0 else ""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0 and (api is None or d["hostapi"] == api):
            out.append({"id": i, "name": d["name"],
                        "default": bool(default_name) and d["name"].startswith(default_name[:20])})
    return out


def find_device(name):
    """Output device by index or (case-insensitive) name substring."""
    if str(name).isdigit():
        return int(name)
    hits = [d["id"] for d in output_devices() if str(name).lower() in d["name"].lower()]
    if not hits:
        raise SystemExit(f"output device '{name}' not found; run with --list-audio")
    return hits[0]


def list_audio():
    for d in output_devices():
        print(f"{'*' if d['default'] else ' '} {d['id']:3d}  {d['name']}")
    print("(* = Windows default output)")


class DeviceOut:
    """One sound output with its own stream, so voice and vibration can play on two devices at once."""

    def __init__(self, device):
        import sounddevice as sd
        self.sd, self.device = sd, int(device)
        info = sd.query_devices(self.device)
        self.name, self.rate = info["name"], int(info["default_samplerate"])
        self.channels = min(2, int(info["max_output_channels"]))
        self._run_id, self._end, self.error = 0, 0.0, None

    @property
    def is_playing(self):
        return time.time() < self._end

    def play(self, wav, sr, gain=1.0, hand="both", loop=1, delay=0.0):
        w = np.clip(np.tile(np.asarray(wav, dtype=np.float32), max(1, int(loop))) * gain, -1, 1)
        if sr != self.rate:   # linear resampling is enough for vibration and speech playback
            w = np.interp(np.arange(int(len(w) * self.rate / sr)) * sr / self.rate, np.arange(len(w)), w).astype(np.float32)
        out = np.repeat(w[:, None], self.channels, axis=1)
        if self.channels == 2 and hand in ("left", "right"):
            out[:, 1 if hand == "left" else 0] = 0
        self._run_id += 1
        rid = self._run_id
        self._end = time.time() + delay + len(w) / self.rate
        threading.Thread(target=self._run, args=(out, rid, delay), daemon=True).start()

    def _run(self, out, rid, delay):
        if delay > 0:
            time.sleep(delay)
        if rid != self._run_id:
            return
        try:
            with self.sd.OutputStream(device=self.device, samplerate=self.rate, channels=self.channels, dtype="float32") as st:
                for k in range(0, len(out), 1024):
                    if rid != self._run_id:   # stopped or replaced
                        break
                    st.write(out[k:k + 1024])
            self.error = None
        except Exception as exc:  # noqa: BLE001  reported to the web page through the status message
            self.error = f"{self.name}: {type(exc).__name__}: {exc}"
            self._end = 0.0
            print(f"[player] playback error on {self.error}")

    def stop(self):
        self._run_id += 1
        self._end = 0.0


# ---------------------------------------------------------------- agent
class Agent:
    def __init__(self, vib=None):
        cfg = {}
        try:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.quest = None
        self.vib_target = vib if vib is not None else cfg.get("vib", "quest")   # "quest" | device id
        self.voice_target = cfg.get("voice")                                     # device id | None (= default)
        listed = {d["id"] for d in output_devices()}   # ids saved under another host API are not valid any more
        if self.vib_target != "quest" and self.vib_target not in listed:
            self.vib_target = "quest"
        if self.voice_target is not None and self.voice_target not in listed:
            self.voice_target = None
        self.outs = {}
        self.apply()

    def out(self, dev):
        import sounddevice as sd
        dev = int(sd.default.device[1] if dev is None else dev)
        if dev not in self.outs:
            self.outs[dev] = DeviceOut(dev)
        return self.outs[dev]

    def apply(self):
        if self.vib_target == "quest" and self.quest is None:
            try:
                from quest_haptic_play import QuestHapticPlayer
                self.quest = QuestHapticPlayer().start()
            except Exception as exc:  # noqa: BLE001  pyopenxr missing / no Link: sound outputs still work
                print(f"[player] Quest unavailable ({exc}); choose a sound output for vibration")
        CONFIG.write_text(json.dumps({"vib": self.vib_target, "voice": self.voice_target}), encoding="utf-8")

    def status(self):
        if self.vib_target == "quest":
            st = self.quest.status if self.quest else "Quest 사용 불가 (pyopenxr 또는 Link 없음)"
            ready = bool(self.quest) and (st.startswith("준비됨") or st == "재생 중")
            playing = bool(self.quest) and self.quest.is_playing
        else:
            try:
                o = self.out(self.vib_target)
                st, ready, playing = f"준비됨 (진동: {o.name})", True, o.is_playing
            except Exception as exc:  # noqa: BLE001
                st, ready, playing = f"진동 장치 오류: {exc}", False, False
        errors = [o.error for o in self.outs.values() if o.error]
        return {"type": "status", "status": st, "connected": ready, "ready": ready, "playing": playing,
                "error": "; ".join(errors) or None,
                "devices": output_devices(), "config": {"vib": self.vib_target, "voice": self.voice_target}}

    def vibrate(self, w, sr, gain, hand, loop=1):
        if self.vib_target == "quest":
            if self.quest:
                self.quest.play(w, sr, hand, gain, loop)
        else:
            self.out(self.vib_target).play(w, sr, gain, hand, loop)

    def stop(self):
        if self.quest:
            self.quest.stop()
        for o in self.outs.values():
            o.stop()


def fetch(base, path):
    data = urllib.request.urlopen(base + path, timeout=30).read()
    w, sr = sf.read(io.BytesIO(data), dtype="float32")
    return (w.mean(axis=1) if w.ndim > 1 else w), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", help="e.g. http://desktop-nst9a4u:7860")
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--audio-device", help="vibration to this sound output (index or name part) instead of the Quest")
    ap.add_argument("--list-audio", action="store_true", help="list sound outputs and exit")
    a = ap.parse_args()
    if a.list_audio:
        list_audio()
        return
    if not a.server:
        ap.error("--server is required")
    base = ipv4_base(a.server.rstrip("/"))
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/player?name={a.name}"
    agent = Agent(vib=find_device(a.audio_device) if a.audio_device else None)
    print(f"[player] {agent.status()['status']}")

    while True:
        try:
            with connect(ws_url, open_timeout=10) as ws:
                print(f"[player] connected to {base} as '{a.name}'. Choose devices on the web page.")
                alive = threading.Event()
                alive.set()

                def report():
                    while alive.is_set():
                        try:
                            ws.send(json.dumps(agent.status()))
                        except Exception:  # noqa: BLE001  connection closed; the outer loop reconnects
                            return
                        time.sleep(1)

                threading.Thread(target=report, daemon=True).start()
                try:
                    for raw in ws:
                        msg = json.loads(raw)
                        cmd = msg.get("cmd")
                        if cmd == "play":   # vibration only
                            w, sr = fetch(base, msg["path"])
                            agent.vibrate(w, sr, msg.get("gain", 1.0), msg.get("hand", "both"), msg.get("loop", 1))
                            print(f"[player] vibration {len(w) / sr:.2f}s")
                        elif cmd == "play_pair":   # narration + vibration, started together
                            voice = fetch(base, msg["voice"])
                            vib = fetch(base, msg["vib"]) if msg.get("vib") else None
                            agent.stop()
                            agent.out(agent.voice_target).play(voice[0], voice[1], msg.get("voice_gain", 1.0),
                                                               delay=max(0.0, msg.get("voice_delay_ms", 0) / 1000))
                            if vib:
                                agent.vibrate(vib[0], vib[1], msg.get("gain", 1.0), msg.get("hand", "both"))
                            print(f"[player] voice {len(voice[0]) / voice[1]:.2f}s" + (" + vibration" if vib else ""))
                        elif cmd == "config":
                            if "vib" in msg:
                                agent.vib_target = "quest" if msg["vib"] == "quest" else int(msg["vib"])
                            if "voice" in msg:
                                agent.voice_target = None if msg["voice"] in (None, "") else int(msg["voice"])
                            agent.apply()
                            print(f"[player] devices: vibration={agent.vib_target} voice={agent.voice_target or 'default'}")
                        elif cmd == "stop":
                            agent.stop()
                finally:
                    alive.clear()
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001  server restarting / network down
            print(f"[player] disconnected ({type(exc).__name__}: {exc}); retrying in 3 s")
            time.sleep(3)
    agent.stop()


if __name__ == "__main__":
    main()
