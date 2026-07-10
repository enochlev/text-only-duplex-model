"""Duplex pipeline test runner, with drift/level debugging built in.

Pipeline:  mic (device rate) -> resample to 16k -> [hush] -> duplex server -> misty

Run it from this directory with uv (see README.md):

    uv sync            # once
    uv run test.py

Server side (the vLLM box, 192.168.0.179):
    vllm serve xinrongzhang2022/MiniCPM-duplex --served-model-name cpm-text-duplex \
        --max-model-len 3000 --port 8555 --trust-remote-code
    python server.py --host 0.0.0.0 --port 8998 --vllm-port 8555 --cpm

Why 16k end-to-end (NOT 24k):
  * The server resamples to 16k for ASR anyway (ASR_SAMPLE_RATE=16000); sending
    24k is pure extra bandwidth plus a second lossy resample.
  * More important: if the module that talks to the server ever labels the
    audio with the wrong sample_rate, 24k data played/consumed as 16k is 1.5x
    slowed down ("low and stretched") and each 2s wall-clock block accumulates
    3s of audio -- the ASR timestamps overrun the block windows and the agent
    goes deaf after a few seconds. With 16k everywhere, a wrong label can't
    hurt because everything IS 16k.

Checklist for the module that talks to the server (remote_duplex):
  * send encoding "pcm_f32le", values normalized to +/-1.0  (the server does
    NOT normalize int16 -- pcm_s16le audio arrives 32768x too hot for ASR)
  * label "sample_rate" with the ACTUAL rate of the bytes (16000 here)
  * websocket needs max_size=None (server snapshots embed base64 WAVs > 1MiB)
  * ignore "snapshot" messages; auto-reconnect when the gradio tunnel drops

How to read the [tap:*] lines printed every 2s:
  * audio_s should track wall_s at every tap. drift = audio_s - wall_s.
  * drift at tap:mic16k should sit near 0. If it goes negative here, the mic
    itself is stalling (device/rate problem).
  * drift at tap:post_hush should sit near -buffer_seconds (a constant lag is
    fine). If it keeps growing more negative, hush is slower than realtime or
    dropping audio -> THIS is what makes the duplex deaf after ~5 seconds.
  * rms tells you if there is actual signal. rms ~0.000x at post_hush while
    mic16k shows rms > 0.01 means the filter is killing the signal.
  * tap:duplex_out prints the rate the server audio comes back at -- the
    speaker's sample_rate MUST match that number or playback is slow/fast.

Each tap also records a WAV into ./debug_wavs/ so you can listen afterwards.
"""
import os, sys
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np

# Local packages (retico_minicpm / retico_mistyrobot / retico_hush) live next
# to this file; retico_core comes from the uv environment (PyPI).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import retico_core
from retico_core import abstract
from retico_core.audio import MicrophoneModule, AudioIU
from retico_mistyrobot.misty_speaker import MistySpeakerModule
from retico_minicpm.remote_duplex import MiniCPMDuplexModule

TARGET_RATE = 16000          # everything after the resampler runs at this
SERVER_URL = os.getenv("DUPLEX_SERVER_URL", "ws://192.168.0.179:8998")
MISTY_IP = os.getenv("MISTY_IP", "192.168.0.156")
HUSH_CHECKPOINT = os.getenv("HUSH_CHECKPOINT", "weya-ai/hush")


class DebugTap(abstract.AbstractConsumingModule):
    """Passive tap: subscribe it anywhere audio flows. Every ~2s it prints
    wall time vs audio time (drift), RMS level, and the stream's rate; it
    also writes everything it sees to debug_wavs/<label>.wav."""

    @staticmethod
    def name(): return "Debug Tap"
    @staticmethod
    def description(): return "Prints drift/level stats and records the stream."
    @staticmethod
    def input_ius(): return [AudioIU]

    def __init__(self, label, report_every=2.0, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.report_every = report_every
        self.rate = None
        self.t0 = None
        self.audio_bytes = 0
        self.sq_sum = 0.0
        self.sq_n = 0
        self.next_report = 0.0
        self.wavfile = None

    def process_update(self, update_message):
        if not update_message:
            return
        now = time.monotonic()
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            raw = getattr(iu, "raw_audio", None) or getattr(iu, "payload", None)
            if not raw:
                continue
            if self.t0 is None:
                self.t0 = now
                self.next_report = now + self.report_every
                self.rate = int(getattr(iu, "rate", 0) or 0)
                os.makedirs("debug_wavs", exist_ok=True)
                self.wavfile = wave.open(f"debug_wavs/{self.label}.wav", "wb")
                self.wavfile.setnchannels(1)
                self.wavfile.setsampwidth(2)
                self.wavfile.setframerate(self.rate or TARGET_RATE)
                print(f"[tap:{self.label}] first audio, rate={self.rate}")
            self.audio_bytes += len(raw)
            self.wavfile.writeframes(raw)
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            self.sq_sum += float(np.sum(x * x))
            self.sq_n += len(x)

        if self.t0 is not None and now >= self.next_report:
            self.next_report = now + self.report_every
            wall_s = now - self.t0
            audio_s = self.audio_bytes / 2 / (self.rate or TARGET_RATE)
            rms = (self.sq_sum / self.sq_n) ** 0.5 if self.sq_n else 0.0
            self.sq_sum, self.sq_n = 0.0, 0
            print(f"[tap:{self.label}] wall={wall_s:6.1f}s audio={audio_s:6.1f}s "
                  f"drift={audio_s - wall_s:+5.1f}s rms={rms:.4f} rate={self.rate}")

    def shutdown(self):
        if self.t0 is not None:
            wall_s = time.monotonic() - self.t0
            audio_s = self.audio_bytes / 2 / (self.rate or TARGET_RATE)
            print(f"[tap:{self.label}] FINAL wall={wall_s:.1f}s audio={audio_s:.1f}s "
                  f"drift={audio_s - wall_s:+.1f}s -> debug_wavs/{self.label}.wav")
        if self.wavfile is not None:
            self.wavfile.close()
            self.wavfile = None


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


# --- hush (optional) -----------------------------------------------------
# Hush runs via the prebuilt libweya_nc shared library (ONNX Runtime inside,
# no torch) from the Hush repo clone next to the training repo. The model
# bundle comes from HUSH_CHECKPOINT: an HF repo id (default weya-ai/hush,
# cached via huggingface-hub -> `uv sync --extra hush`) or a local path.
# NOTE: Hush keeps the DOMINANT speaker — it removes typing/background
# talkers while YOU speak, but it does NOT remove Misty self-feedback (when
# you are silent, Misty is the dominant speaker and passes through).
hush_filter = None
if HUSH_CHECKPOINT.strip().lower() not in {"", "0", "off", "none", "disabled"}:
    try:
        from retico_hush.hush import HushFilterModule
        hush_filter = HushFilterModule(
            checkpoint_path=HUSH_CHECKPOINT,
            atten_lim_db=float(os.getenv("HUSH_ATTEN_LIM_DB", "100")),
            input_gain=float(os.getenv("HUSH_INPUT_GAIN", "1.0")),
            # zero out sub-threshold output frames: hush's faint residue on
            # gated audio (e.g. whisper-quiet Misty feedback) is still
            # transcribable by the ASR. Measured margins 2026-07-10: residue
            # p90=0.0014 vs user-speech p10=0.0068 -> 0.003 splits them.
            gate_rms=float(os.getenv("HUSH_GATE_RMS", "0.003")),
        )
        print(f"[test.py] hush filter enabled ({HUSH_CHECKPOINT})")
    except Exception as exc:
        print(f"[test.py] hush unavailable ({type(exc).__name__}: {exc}); running without filter")
else:
    print(f"[test.py] hush disabled (HUSH_CHECKPOINT={HUSH_CHECKPOINT!r})")

# --- microphone ---------------------------------------------------------
# NOTE: retico-core's MicrophoneModule signature is (frame_length, rate,
# sample_width). It has NO sample_rate/device_index params -- unknown kwargs
# are silently swallowed, so MicrophoneModule(sample_rate=48000) actually
# opens at the 44100 DEFAULT. Always pass rate=.
import pyaudio
p = pyaudio.PyAudio()
try:
    device_rate = int(round(p.get_default_input_device_info().get("defaultSampleRate", 16000)))
except Exception:
    device_rate = 16000
finally:
    p.terminate()

# Try to open the mic at 16k directly (most devices allow it); fall back to
# the device's native rate and let the resampler do the conversion.
try:
    mic = MicrophoneModule(rate=TARGET_RATE, frame_length=0.02)
    _probe = mic._p.open(rate=TARGET_RATE, channels=1, format=pyaudio.paInt16,
                         input=True, frames_per_buffer=int(TARGET_RATE * 0.02))
    _probe.close()
    print(f"[test.py] mic at {TARGET_RATE} Hz")
except Exception as exc:
    print(f"[test.py] mic can't open at {TARGET_RATE} ({exc}); using device rate {device_rate}")
    mic = MicrophoneModule(rate=device_rate, frame_length=0.02)

resampler = ResampleModule(target_rate=TARGET_RATE)
duplex = MiniCPMDuplexModule(server_url=SERVER_URL)
# The speaker adopts the actual stream rate from the first IU it receives
# ([tap:duplex_out] prints it); 16000 is just the starting value.
speaker = MistySpeakerModule(ip=MISTY_IP, sample_rate=16000, volume=20)

tap_mic = DebugTap("mic16k")        # after resampler: the pipeline's clock
tap_hush = DebugTap("post_hush")    # what actually reaches the duplex
tap_out = DebugTap("duplex_out")    # what the server sends back

print(f"[test.py] duplex server: {SERVER_URL}")
print(f"[test.py] misty robot  : {MISTY_IP}")

mic.subscribe(resampler)
resampler.subscribe(tap_mic)
if hush_filter is not None:
    print("HUSH ENABLED: mic -> resample -> hush -> duplex -> misty")
    resampler.subscribe(hush_filter)
    hush_filter.subscribe(tap_hush)
    hush_filter.subscribe(duplex)
else:
    resampler.subscribe(tap_hush)
    resampler.subscribe(duplex)
duplex.subscribe(tap_out)
duplex.subscribe(speaker)

modules = [tap_mic, tap_hush, tap_out, resampler, duplex, speaker, mic]
if hush_filter is not None:
    modules.insert(3, hush_filter)

for m in modules:
    m.run()

print("Running... speak, then press Enter to stop.")
input()

for m in reversed(modules):
    m.stop()
