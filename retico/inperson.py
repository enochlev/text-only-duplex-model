"""In-person survey client: PC mic -> resample 16k -> [hush] -> duplex server -> Misty.

Runs on the intern's PC next to the survey wizard (`python run_demo.py --inperson ...`,
same machine). The wizard controls WHICH blinded system slot is live:

  1. This script polls  GET {SURVEY_URL}/inperson_target  until the wizard activates a
     slot (participant reached a talk step) — the response carries that slot's server URL.
  2. It then builds the full audio pipeline (same as test.py) against that URL and, once a
     second, relays the latest transcript snapshot to  POST {SURVEY_URL}/live_snapshot
     (audio data-URIs stripped) so the participant sees their conversation live.
  3. When the wizard deactivates the slot (participant clicked "I'm done" / 5-min timer),
     the POST response's active_slot changes — the pipeline is torn down and we go back
     to polling. One launch covers the whole session (both systems).

Run from this directory (see README.md for the uv setup):

    uv run inperson.py

Env (.env or shell):
    SURVEY_URL   http://127.0.0.1:7870   the local run_demo.py --inperson wizard
    MISTY_IP     192.168.0.102           Misty robot REST API
    HUSH_CHECKPOINT  weya-ai/hush        set to "off" to disable the hush filter
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import retico_core
from retico_core import abstract
from retico_core.audio import MicrophoneModule, AudioIU
from retico_mistyrobot.misty_speaker import MistySpeakerModule
from retico_minicpm.remote_duplex import MiniCPMDuplexModule

TARGET_RATE = 16000  # everything after the resampler runs at this (see test.py's rationale)

SURVEY_URL = os.getenv("SURVEY_URL", "http://127.0.0.1:7870").rstrip("/")
MISTY_IP = os.getenv("MISTY_IP", "192.168.0.102")
HUSH_CHECKPOINT = os.getenv("HUSH_CHECKPOINT", "weya-ai/hush")
POLL_S = 2.0          # idle poll of /inperson_target
SNAPSHOT_S = 1.0      # transcript relay cadence while a slot is live


# Same module as test.py's — duplicated here because test.py is a top-level script
# (importing it would run its whole pipeline, ending at input()).
class ResampleModule(abstract.AbstractModule):
    """Resample incoming audio to target_rate, sample-count exact."""

    @staticmethod
    def name(): return "Resample Module"
    @staticmethod
    def description(): return "Resamples audio to the target sample rate."
    @staticmethod
    def input_ius(): return [AudioIU]
    @staticmethod
    def output_iu(): return AudioIU

    def __init__(self, target_rate=TARGET_RATE, **kwargs):
        super().__init__(**kwargs)
        self.target_rate = target_rate

    def process_update(self, update_message):
        output_message = None
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            raw_audio = iu.raw_audio
            if raw_audio is None or len(raw_audio) == 0:
                continue
            audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            sr_in = iu.rate
            if sr_in != self.target_rate:
                n_out = max(1, int(round(audio.size * self.target_rate / sr_in)))
                xp = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
                x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
                resampled = np.interp(x, xp, audio).astype(np.float32)
            else:
                resampled = audio
            pcm16 = (resampled * 32767.0).clip(-32768, 32767).astype(np.int16)
            out_iu = self.create_iu(iu)
            out_iu.set_audio(pcm16.tobytes(), len(pcm16), self.target_rate, iu.sample_width)
            if output_message is None:
                output_message = retico_core.UpdateMessage.from_iu(out_iu, ut)
            else:
                output_message.add_iu(out_iu, ut)
        return output_message


def build_hush():
    if HUSH_CHECKPOINT.strip().lower() in {"", "0", "off", "none", "disabled"}:
        print(f"[inperson] hush disabled (HUSH_CHECKPOINT={HUSH_CHECKPOINT!r})")
        return None
    try:
        from retico_hush.hush import HushFilterModule
        module = HushFilterModule(
            checkpoint_path=HUSH_CHECKPOINT,
            atten_lim_db=float(os.getenv("HUSH_ATTEN_LIM_DB", "100")),
            input_gain=float(os.getenv("HUSH_INPUT_GAIN", "1.0")),
            gate_rms=float(os.getenv("HUSH_GATE_RMS", "0.003")),
        )
        print(f"[inperson] hush filter enabled ({HUSH_CHECKPOINT})")
        return module
    except Exception as exc:
        print(f"[inperson] hush unavailable ({type(exc).__name__}: {exc}); running without filter")
        return None


def build_mic():
    """Same probe logic as test.py: prefer 16 k, fall back to the device rate."""
    import pyaudio
    p = pyaudio.PyAudio()
    try:
        device_rate = int(round(p.get_default_input_device_info().get("defaultSampleRate", 16000)))
    except Exception:
        device_rate = 16000
    finally:
        p.terminate()
    try:
        mic = MicrophoneModule(rate=TARGET_RATE, frame_length=0.02)
        probe = mic._p.open(rate=TARGET_RATE, channels=1, format=pyaudio.paInt16,
                            input=True, frames_per_buffer=int(TARGET_RATE * 0.02))
        probe.close()
        print(f"[inperson] mic at {TARGET_RATE} Hz")
        return mic
    except Exception as exc:
        print(f"[inperson] mic can't open at {TARGET_RATE} ({exc}); using device rate {device_rate}")
        return MicrophoneModule(rate=device_rate, frame_length=0.02)


def snapshot_payload(duplex):
    """Latest transcript snapshot as a dict, with the heavy audio data-URIs stripped."""
    client = duplex.client
    snap = client.get_latest_snapshot() if client is not None else None
    if snap is None:
        return None
    d = snap.to_dict()
    for block in d.get("blocks", []) + ([d["current_block"]] if d.get("current_block") else []):
        block.pop("mic_audio_uri", None)
        block.pop("tts_audio_uri", None)
    return d


def run_slot(slot: str, url: str):
    """Run the mic→duplex→Misty pipeline for one activation; return when deactivated."""
    print(f"[inperson] === slot {slot} ACTIVE → {url} ===")
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("debug_wavs", exist_ok=True)

    mic = build_mic()
    resampler = ResampleModule(target_rate=TARGET_RATE)
    hush = build_hush()
    duplex = MiniCPMDuplexModule(server_url=url,
                                 wav_path=f"debug_wavs/inperson_{ts}_{slot}.wav",
                                 lite_snapshots=True)  # text-only snapshots: 30x less tunnel traffic
    speaker = MistySpeakerModule(ip=MISTY_IP, sample_rate=16000, volume=20)

    mic.subscribe(resampler)
    if hush is not None:
        resampler.subscribe(hush)
        hush.subscribe(duplex)
    else:
        resampler.subscribe(duplex)
    duplex.subscribe(speaker)

    modules = [resampler, duplex, speaker, mic] + ([hush] if hush is not None else [])
    for m in modules:
        m.run()

    def relay(extra=None):
        """POST heartbeat/snapshot; returns the wizard's current active_slot ('?' if unreachable)."""
        connected = bool(duplex.client is not None and duplex.client.connected)
        payload = {"snapshot": snapshot_payload(duplex), "connected": connected}
        if extra:
            payload.update(extra)
        try:
            return requests.post(f"{SURVEY_URL}/live_snapshot", json=payload, timeout=3).json().get("active_slot")
        except Exception as exc:
            print(f"[inperson] wizard unreachable ({exc}); retrying")
            return "?"

    try:
        # Wait for the duplex WS to actually connect. prepare_run() raises inside a
        # retico thread (e.g. a transient tunnel 404), which would otherwise leave a
        # zombie pipeline and a wizard stuck on "connecting". Timeout → teardown →
        # the outer loop re-activates us while the slot is still live (auto-retry).
        deadline = time.time() + 25
        while not (duplex.client is not None and duplex.client.connected):
            if time.time() > deadline:
                print(f"[inperson] could not connect to {url} within 25s — will retry")
                relay({"error": "robot could not reach the model server — retrying"})
                return
            active = relay()
            if active not in (slot, "?"):
                print(f"[inperson] slot {slot} deactivated while connecting")
                return
            time.sleep(1)

        was_connected = True
        while True:
            time.sleep(SNAPSHOT_S)
            connected = bool(duplex.client is not None and duplex.client.connected)
            if was_connected and not connected:
                # tunnel/server dropped mid-session: tear down and let the outer
                # loop reconnect fresh (the wizard keeps the slot active)
                print("[inperson] connection lost — tearing down to reconnect")
                relay({"error": "robot lost the model connection — reconnecting"})
                return
            active = relay()
            if active not in (slot, "?"):
                print(f"[inperson] slot {slot} deactivated (now {active})")
                return
    finally:
        for m in reversed(modules):
            try:
                m.stop()
            except Exception as exc:
                print(f"[inperson] {m.name()} stop failed: {exc}")
        print(f"[inperson] === slot {slot} torn down (wav: debug_wavs/inperson_{ts}_{slot}.wav) ===")


def main():
    print(f"[inperson] survey wizard : {SURVEY_URL}")
    print(f"[inperson] misty robot   : {MISTY_IP}")
    print("[inperson] waiting for the wizard to activate a system slot… (Ctrl-C to quit)")
    while True:
        try:
            t = requests.get(f"{SURVEY_URL}/inperson_target", timeout=3).json()
        except Exception:
            time.sleep(POLL_S)
            continue
        slot, url = t.get("active_slot"), t.get("url")
        if slot and url:
            run_slot(slot, url)
            print("[inperson] waiting for the next activation…")
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[inperson] bye")
