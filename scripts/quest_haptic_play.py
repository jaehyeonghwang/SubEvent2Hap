"""Play vibration waveforms on Meta Quest Touch controllers via OpenXR (Quest Link / Air Link).

Uses XR_FB_haptic_pcm so the full waveform (not just an amplitude envelope) is sent
to the controller's wideband actuator. The headset must be in Link mode.

`QuestHapticPlayer` keeps an OpenXR session alive on a background thread (reconnecting
when Link comes and goes) so playback requests start immediately.

Usage:
    python scripts/quest_haptic_play.py out/foo.wav [--hand both|left|right] [--gain 1.0] [--loop 1]
"""
import argparse
import ctypes
import queue
import threading
import time
from ctypes import POINTER, byref, c_float, c_uint32, c_void_p, cast

import numpy as np
import soundfile as sf
import xr
from xr.platform.windows import _LUID

# ---------------------------------------------------------------- D3D11 device
_d3d11 = ctypes.WinDLL("d3d11")
_dxgi = ctypes.WinDLL("dxgi")


class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))


class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_ubyte * 8)]


def _vcall(obj, index, restype, *argtypes):
    vtbl = cast(cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[index])


def _find_adapter(luid: bytes):
    factory = c_void_p()
    if _dxgi.CreateDXGIFactory1(byref(_IID_IDXGIFactory1), byref(factory)) != 0:
        return None, None
    i = 0
    while True:
        adapter = c_void_p()
        # IDXGIFactory1::EnumAdapters1 (vtable 12)
        if _vcall(factory, 12, ctypes.c_long, ctypes.c_uint, POINTER(c_void_p))(factory, i, byref(adapter)) != 0:
            return None, None
        desc = _DXGI_ADAPTER_DESC()
        _vcall(adapter, 8, ctypes.c_long, POINTER(_DXGI_ADAPTER_DESC))(adapter, byref(desc))  # GetDesc
        if bytes(desc.AdapterLuid) == luid:
            return adapter, desc.Description
        i += 1


def create_d3d11_device(luid: bytes):
    adapter, name = _find_adapter(luid)
    device = c_void_p()
    levels = (ctypes.c_int * 2)(0xB100, 0xB000)
    D3D_DRIVER_TYPE_UNKNOWN, D3D_DRIVER_TYPE_HARDWARE = 0, 1
    hr = _d3d11.D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN if adapter else D3D_DRIVER_TYPE_HARDWARE,
                                  None, 0, levels, 2, 7, byref(device), None, None)
    if hr != 0:
        raise RuntimeError(f"D3D11CreateDevice failed: 0x{hr & 0xFFFFFFFF:08X}")
    return device, name


# ---------------------------------------------------------------- OpenXR helpers
def ext_fn(instance, name, pfn_type):
    return cast(xr.get_instance_proc_addr(instance, name), pfn_type)


def check(result, what):
    r = xr.Result(result)
    if r.value < 0:
        raise RuntimeError(f"{what} failed: {r.name}")


_UNSET = 0xFFFFFFFF
PACED_CHUNK_SEC = 0.5   # chunk size when the runtime gives no samplesConsumed feedback
PACED_LEAD_SEC = 0.15   # append the next chunk this long before the queued audio runs out


class LinkUnavailable(Exception):
    pass


class _Session:
    """One OpenXR instance + session with a haptic output action for each hand."""

    def __init__(self):
        self.instance = xr.create_instance(xr.InstanceCreateInfo(
            application_info=xr.ApplicationInfo(application_name="hapticgen_player", engine_name="pyopenxr",
                                                api_version=xr.Version(1, 0, 0)),
            enabled_extension_names=[xr.KHR_D3D11_ENABLE_EXTENSION_NAME, xr.FB_HAPTIC_PCM_EXTENSION_NAME],
        ))
        self.session = None
        try:
            try:
                system_id = xr.get_system(self.instance, xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY))
            except xr.exception.FormFactorUnavailableError:
                raise LinkUnavailable()
            # pyopenxr's default adapter_luid=0 fails ctypes type checks; pass an explicit _LUID
            reqs = xr.GraphicsRequirementsD3D11KHR(adapter_luid=_LUID())
            check(ext_fn(self.instance, "xrGetD3D11GraphicsRequirementsKHR", xr.PFN_xrGetD3D11GraphicsRequirementsKHR)(
                self.instance, system_id, byref(reqs)), "xrGetD3D11GraphicsRequirementsKHR")
            self.device, self.gpu_name = create_d3d11_device(bytes(reqs.adapter_luid))
            binding = xr.GraphicsBindingD3D11KHR(device=cast(self.device, POINTER(ctypes.c_long)))
            self.session = xr.create_session(self.instance, xr.SessionCreateInfo(
                system_id=system_id, next=cast(byref(binding), c_void_p)))

            self.hands = {"left": xr.string_to_path(self.instance, "/user/hand/left"),
                          "right": xr.string_to_path(self.instance, "/user/hand/right")}
            self.action_set = xr.create_action_set(self.instance, xr.ActionSetCreateInfo(
                action_set_name="haptics", localized_action_set_name="Haptics"))
            self.haptic = xr.create_action(self.action_set, xr.ActionCreateInfo(
                action_type=xr.ActionType.VIBRATION_OUTPUT, action_name="vibrate", localized_action_name="Vibrate",
                subaction_paths=list(self.hands.values())))
            xr.suggest_interaction_profile_bindings(self.instance, xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(self.instance, "/interaction_profiles/oculus/touch_controller"),
                suggested_bindings=[xr.ActionSuggestedBinding(
                    self.haptic, xr.string_to_path(self.instance, f"/user/hand/{h}/output/haptic")) for h in self.hands]))
            xr.attach_session_action_sets(self.session, xr.SessionActionSetsAttachInfo(action_sets=[self.action_set]))
            self._get_rate = ext_fn(self.instance, "xrGetDeviceSampleRateFB", xr.PFN_xrGetDeviceSampleRateFB)
        except BaseException:
            self.close()
            raise
        self.state = xr.SessionState.UNKNOWN
        self.running = False

    def close(self):
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def poll(self):
        """Process pending events. Returns False when the session is lost/exiting."""
        while True:
            try:
                ev = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return True
            t = xr.StructureType(ev.type)
            if t == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                self.state = xr.SessionState(cast(byref(ev), POINTER(xr.EventDataSessionStateChanged)).contents.state)
                if self.state == xr.SessionState.READY:
                    xr.begin_session(self.session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
                    self.running = True
                elif self.state == xr.SessionState.STOPPING:
                    xr.end_session(self.session)
                    self.running = False
                elif self.state in (xr.SessionState.EXITING, xr.SessionState.LOSS_PENDING):
                    return False
            elif t == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                return False

    def frame(self):
        """Submit an empty frame (keeps the session alive / focused)."""
        fs = xr.wait_frame(self.session)
        xr.begin_frame(self.session)
        xr.end_frame(self.session, xr.FrameEndInfo(display_time=fs.predicted_display_time,
                                                   environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE))
        if self.state == xr.SessionState.FOCUSED:
            xr.sync_actions(self.session, xr.ActionsSyncInfo(
                active_action_sets=[xr.ActiveActionSet(self.action_set, xr.NULL_PATH)]))

    @property
    def focused(self):
        return self.state == xr.SessionState.FOCUSED

    def info(self, hand):
        return xr.HapticActionInfo(action=self.haptic, subaction_path=self.hands[hand])

    def sample_rate(self, hand):
        rs = xr.DevicePcmSampleRateStateFB()
        check(self._get_rate(self.session, byref(self.info(hand)), byref(rs)), "xrGetDeviceSampleRateFB")
        return rs.sample_rate or 2000.0

    def push_pcm(self, hand, chunk: np.ndarray, sr: float, append: bool):
        """Returns samples consumed, or None if the runtime didn't report it (Link PC runtime doesn't)."""
        consumed = c_uint32(_UNSET)
        pcm = xr.HapticPcmVibrationFB(
            buffer_size=len(chunk), buffer=chunk.ctypes.data_as(POINTER(c_float)), sample_rate=sr,
            append=xr.TRUE if append else xr.FALSE, samples_consumed=ctypes.pointer(consumed))
        xr.apply_haptic_feedback(self.session, self.info(hand), cast(byref(pcm), POINTER(xr.HapticBaseHeader)).contents)
        return None if consumed.value == _UNSET else consumed.value

    def stop_haptics(self, hand):
        xr.stop_haptic_feedback(self.session, self.info(hand))


def _hands(hand):
    return ["left", "right"] if hand == "both" else [hand]


def _resample(wav, src_sr, dst_sr):
    n_out = int(round(len(wav) * dst_sr / src_sr))
    return np.interp(np.arange(n_out) * src_sr / dst_sr, np.arange(len(wav)), wav).astype(np.float32)


class QuestHapticPlayer:
    """Background OpenXR session that plays PCM haptics on request."""

    def __init__(self, reconnect_interval=2.0):
        self.reconnect_interval = reconnect_interval
        self.status = "시작 전"
        self.connected = False
        self.device_rate = None
        self._cmds = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, name="quest-haptics", daemon=True)
        self._playing = []  # [{hand, buf, sr, pos}]
        self._play_end = 0.0

    def start(self):
        self._thread.start()
        return self

    def close(self):
        self._stop_evt.set()
        self._thread.join(timeout=5)

    def play(self, wav: np.ndarray, sr: int, hand="both", gain=1.0, loop=1):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        self._cmds.put(("play", np.clip(wav * gain, -1.0, 1.0), sr, hand, max(1, int(loop))))

    def stop(self):
        self._cmds.put(("stop",))

    @property
    def is_playing(self):
        return bool(self._playing) or time.time() < self._play_end

    # ---------------------------------------------------------------- thread
    def _run(self):
        while not self._stop_evt.is_set():
            try:
                self.status = "Link 연결 시도 중…"
                s = _Session()
            except LinkUnavailable:
                self.connected = False
                self.status = "헤드셋 없음 – Quest에서 Link/Air Link를 켜세요"
                self._drain_cmds()
                self._stop_evt.wait(self.reconnect_interval)
                continue
            except Exception as e:  # noqa: BLE001
                self.connected = False
                self.status = f"OpenXR 오류: {e}"
                self._drain_cmds()
                self._stop_evt.wait(self.reconnect_interval)
                continue
            try:
                self._session_loop(s)
            except Exception as e:  # noqa: BLE001
                self.status = f"세션 오류: {e}"
            finally:
                self.connected = False
                self._playing = []
                s.close()
        self.status = "종료됨"

    def _drain_cmds(self):
        try:
            while True:
                self._cmds.get_nowait()
        except queue.Empty:
            pass

    def _session_loop(self, s: _Session):
        self.connected = True
        while not self._stop_evt.is_set():
            if not s.poll():
                self.status = "세션 종료됨 – 재연결 대기"
                return
            if not s.running:
                self.status = f"세션 대기 ({s.state.name})"
                time.sleep(0.01)
                continue
            s.frame()
            if not s.focused:
                self.status = f"연결됨, 포커스 없음 ({s.state.name}) – 헤드셋을 착용하세요"
                self._drain_cmds()
                self._playing = []
                continue
            if self.device_rate is None:
                self.device_rate = s.sample_rate("right")
            self._handle_cmds(s)
            self._pump(s)
            self.status = "재생 중" if self.is_playing else f"준비됨 (컨트롤러 PCM {self.device_rate:.0f} Hz, GPU {s.gpu_name})"

    def _handle_cmds(self, s: _Session):
        try:
            while True:
                cmd = self._cmds.get_nowait()
                if cmd[0] == "stop":
                    for h in s.hands:
                        s.stop_haptics(h)
                    self._playing = []
                    self._play_end = 0.0
                elif cmd[0] == "play":
                    _, wav, sr, hand, loop = cmd
                    for h in s.hands:
                        s.stop_haptics(h)
                    self._playing = []
                    for h in _hands(hand):
                        dsr = s.sample_rate(h)
                        buf = np.ascontiguousarray(np.tile(_resample(wav, sr, dsr), loop))
                        self._playing.append({"hand": h, "buf": buf, "sr": dsr, "pos": 0})
                    self._play_end = time.time() + len(wav) / sr * loop
        except queue.Empty:
            pass

    def _pump(self, s: _Session):
        now = time.time()
        for pl in self._playing:
            remaining = len(pl["buf"]) - pl["pos"]
            if remaining <= 0:
                continue
            paced = pl.get("paced", False)
            if paced and now < pl["queued_until"] - PACED_LEAD_SEC:
                continue
            limit = int(PACED_CHUNK_SEC * pl["sr"]) if paced else xr.MAX_HAPTIC_PCM_BUFFER_SIZE_FB
            n = min(remaining, limit)
            chunk = pl["buf"][pl["pos"]:pl["pos"] + n]
            consumed = s.push_pcm(pl["hand"], chunk, pl["sr"], append=pl["pos"] > 0)
            if consumed is None:
                # No feedback: assume the runtime queued the whole chunk and pace by wall clock.
                consumed = n
                pl["paced"] = True
                pl["queued_until"] = max(now, pl.get("queued_until", now)) + n / pl["sr"]
            pl["pos"] += consumed
        self._playing = [pl for pl in self._playing if pl["pos"] < len(pl["buf"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--hand", choices=["both", "left", "right"], default="both")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--loop", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=15.0, help="seconds to wait for Link/focus")
    args = ap.parse_args()

    wav, sr = sf.read(args.wav, dtype="float32")
    player = QuestHapticPlayer().start()
    t0 = time.time()
    last = None
    while not player.status.startswith("준비됨"):
        if player.status != last:
            print(player.status)
            last = player.status
        if time.time() - t0 > args.timeout:
            player.close()
            raise SystemExit("timed out waiting for Link / focus")
        time.sleep(0.1)
    print(player.status)
    player.play(wav, sr, args.hand, args.gain, args.loop)
    time.sleep(0.2)
    while player.is_playing:
        time.sleep(0.05)
    time.sleep(0.2)
    print("done")
    player.close()


if __name__ == "__main__":
    main()
