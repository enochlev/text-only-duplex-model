"""Hush background-speaker suppression — ONNX runtime edition.

Runs the Hush model through the prebuilt ``libweya_nc`` shared library
(ONNX Runtime inside, ctypes from Python) — the deployment path recommended
by the Hush repo. No torch, no libdf; runs ~50x realtime on an M-series CPU
with 10 ms frames and ~20 ms algorithmic latency.

Requirements
------------
* the Hush repo cloned next to the training repo (provides the shared
  library at ``Hush/deployment/lib/``), or ``HUSH_LIB_PATH`` pointing at it
* the ONNX model bundle, resolved from ``checkpoint_path``:
    - a local ``.tar.gz`` path, or
    - a HuggingFace repo id like ``weya-ai/hush`` (downloaded and cached via
      ``huggingface-hub`` — install with ``uv sync --extra hush``), or
    - a legacy ``.ckpt`` path (the ONNX bundle sitting next to it is used)

Timing contract (same as the old torch version): for every sample in,
exactly one sample out — the module must behave like a wire or the duplex
server's wall-clock block windows drift and the agent goes deaf. The native
lib processes fixed 10 ms frames 1:1, so only a sub-frame remainder is ever
buffered (<10 ms), never accumulated.

KNOWN LIMITATION (measured 2026-07-10 on real recordings): Hush keeps the
DOMINANT foreground speaker. If the user is silent and Misty's playback is
the loudest voice at the mic, Hush treats Misty as the primary speaker and
passes the feedback through (~-1 dB). Hush removes typing/background
speakers while the USER talks; it does not remove robot self-feedback.
"""

import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import retico_core
import retico_core.abstract as abstract
from retico_core.audio import AudioIU

from .weya_nc import WeyaModel

# repo root = text-only-duplex-model/ (this file lives in retico/retico_hush/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ONNX_BUNDLE = "advanced_dfnet16k_model_best_onnx.tar.gz"
_LIB_NAMES = {
    "Darwin": "libweya_nc.dylib",
    "Linux": "libweya_nc.so",
    "Windows": "weya_nc.dll",
}


def _hush_roots():
    return (_REPO_ROOT / "Hush", Path.cwd() / "Hush")


def _find_lib(explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"libweya_nc not found at {p}")
    env = os.getenv("HUSH_LIB_PATH")
    if env and Path(env).exists():
        return Path(env)
    name = _LIB_NAMES.get(platform.system(), "libweya_nc.so")
    for root in _hush_roots():
        p = root / "deployment" / "lib" / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {name}. Clone https://github.com/pulp-vision/Hush next to "
        f"the training repo ({_REPO_ROOT}) or set HUSH_LIB_PATH."
    )


def _resolve_model(spec: str) -> Path:
    """Resolve the ONNX bundle from a local path, HF repo id, or Hush clone."""
    p = Path(spec)
    if p.exists():
        if p.suffix == ".ckpt":
            sibling = p.parent / _ONNX_BUNDLE
            if sibling.exists():
                return sibling
            raise FileNotFoundError(
                f"{p} is a torch checkpoint; this module runs ONNX and needs "
                f"{_ONNX_BUNDLE} next to it (or pass an HF repo id like 'weya-ai/hush')."
            )
        return p
    # HF repo id: exactly one slash, no file suffix (e.g. "weya-ai/hush")
    if spec.count("/") == 1 and not p.suffix:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=spec, filename=f"onnx/{_ONNX_BUNDLE}"))
    for root in _hush_roots():
        c = root / "deployment" / "models" / _ONNX_BUNDLE
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not resolve Hush model from {spec!r}: not a local file, not an HF "
        f"repo id, and no bundle at Hush/deployment/models/{_ONNX_BUNDLE}."
    )


class HushFilterModule(abstract.AbstractModule):
    """Real-time background speaker suppression using Hush (libweya_nc/ONNX).

    Parameters
    ----------
    checkpoint_path : str
        ONNX bundle path, HF repo id (default ``weya-ai/hush``), or legacy
        ``.ckpt`` path (its sibling ONNX bundle is used).
    atten_lim_db : float
        Maximum suppression depth in dB (100 = unlimited, the model default).
        Lower it (e.g. 20) to make the filter gentler.
    input_gain : float
        Linear gain applied before the model. Hush's foreground/background
        decision is level-dependent — a very quiet mic (far-field) gets
        gated entirely. Prefer fixing the real mic level; this is a rescue
        knob.
    gate_rms : float
        Output gate with hangover: while the output stays below this RMS
        (and has for longer than the hangover), frames become true digital
        zeros. Hush attenuates non-foreground audio but leaves a faint
        intelligible residue (~0.001 RMS) that a sensitive ASR can still
        transcribe — e.g. whisper-quiet Misty feedback. True zeros are the
        only thing an ASR provably can't transcribe. The gate opens
        instantly on speech and stays open gate_hang_s after the level
        drops, so quiet word tails are never chopped. Measured margins
        (2026-07-10): residue p90 = 0.0014, user speech p10 = 0.0068, so
        0.003 separates them cleanly. 0 disables.
    gate_hang_s : float
        How long the gate stays open after the level falls below gate_rms
        (default 0.3 s).
    lib_path : str, optional
        Explicit path to the libweya_nc shared library.
    """

    def __init__(self, checkpoint_path="weya-ai/hush", atten_lim_db=100.0,
                 input_gain=1.0, gate_rms=0.0, gate_hang_s=0.3,
                 lib_path=None, **kwargs):
        super().__init__(**kwargs)
        self.atten_lim_db = float(atten_lim_db)
        self.input_gain = float(input_gain)
        self.gate_rms = float(gate_rms)
        self.gate_hang_s = float(gate_hang_s)
        self._gate_hang_left = 0  # frames of hangover remaining

        # Resolve + load in the constructor so a missing lib/model fails
        # fast where callers can catch it and fall back to an unfiltered
        # pipeline (setup() runs inside retico's run() where errors hide).
        self._lib_file = _find_lib(lib_path)
        self._model_file = _resolve_model(str(checkpoint_path))
        self._weya = WeyaModel(lib_path=self._lib_file, model_path=self._model_file)
        print(f"[hush] libweya_nc loaded: {self._lib_file}")
        print(f"[hush] ONNX bundle      : {self._model_file}")

        self._session = None
        self._session_rate = None
        self._in_buffer = np.zeros(0, dtype=np.float32)   # at stream rate
        self._out_bytes = b""                              # int16, sliced into 20ms IUs
        self._stat_in_sq = 0.0
        self._stat_out_sq = 0.0
        self._stat_n = 0
        self._stat_proc_s = 0.0
        self._stat_next_report = 0.0
        self._stat_frames = 0
        self._stat_gated = 0

    @staticmethod
    def name(): return "Hush Filter"

    @staticmethod
    def description(): return "Real-time background speaker suppression (Hush/ONNX)"

    @staticmethod
    def input_ius(): return [AudioIU]

    @staticmethod
    def output_iu(): return AudioIU

    def setup(self):
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._out_bytes = b""

    def _ensure_session(self, rate: int):
        if self._session is not None and self._session_rate == rate:
            return
        if self._session is not None:
            print(f"[hush] WARNING input rate changed {self._session_rate} -> {rate}, resetting")
            self._session.close()
            self._in_buffer = np.zeros(0, dtype=np.float32)
            self._out_bytes = b""
        self._session = self._weya.create_session(
            sample_rate=rate, atten_lim_db=self.atten_lim_db
        )
        self._session_rate = rate
        print(f"[hush] session: rate={rate} frame={self._session.frame_length} samples "
              f"({1000 * self._session.frame_length / rate:.0f} ms), "
              f"atten_lim={self.atten_lim_db} dB, input_gain=x{self.input_gain:g}, "
              f"gate_rms={self.gate_rms:g}")

    def process_update(self, update_message):
        output_msg = None

        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            raw = iu.raw_audio if getattr(iu, "raw_audio", None) is not None else iu.payload
            if raw is None or len(raw) == 0:
                continue

            rate = int(getattr(iu, "rate", 16000) or 16000)
            self._ensure_session(rate)

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if self.input_gain != 1.0:
                audio = np.clip(audio * self.input_gain, -1.0, 1.0)
            self._in_buffer = np.concatenate([self._in_buffer, audio])

            fl = self._session.frame_length
            n_frames = len(self._in_buffer) // fl
            if n_frames == 0:
                continue

            t0 = time.perf_counter()
            outs = []
            for i in range(n_frames):
                outs.append(self._session.process_frame(self._in_buffer[i * fl:(i + 1) * fl]))
            self._stat_proc_s += time.perf_counter() - t0

            processed = np.concatenate(outs)
            self._in_buffer = self._in_buffer[n_frames * fl:]

            self._stat_in_sq += float(np.sum(audio ** 2))
            self._stat_out_sq += float(np.sum(processed ** 2))
            self._stat_n += len(processed)

            self._out_bytes += (
                (processed * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
            )

            # Slice into exact 20ms frames at the stream rate.
            frame_size = int(rate * 0.02) * 2
            sample_width = getattr(iu, "sample_width", 2) or 2
            while len(self._out_bytes) >= frame_size:
                chunk = self._out_bytes[:frame_size]
                self._out_bytes = self._out_bytes[frame_size:]

                self._stat_frames += 1
                if self.gate_rms > 0.0:
                    frame_i16 = np.frombuffer(chunk, dtype=np.int16)
                    frame_rms = float(np.sqrt(np.mean(
                        (frame_i16.astype(np.float32) / 32768.0) ** 2)))
                    hang_frames = int(self.gate_hang_s / 0.02)
                    if frame_rms >= self.gate_rms:
                        self._gate_hang_left = hang_frames  # open instantly
                    elif self._gate_hang_left > 0:
                        self._gate_hang_left -= 1           # hangover: stay open
                    else:
                        # true digital silence: the ASR can't transcribe the
                        # faint residue hush leaves on gated audio
                        chunk = b"\x00" * frame_size
                        self._stat_gated += 1

                new_iu = self.create_iu(iu)
                new_iu.payload = chunk
                new_iu.raw_audio = chunk
                new_iu.rate = rate
                new_iu.sample_width = sample_width
                new_iu.nframes = frame_size // sample_width

                if output_msg is None:
                    output_msg = retico_core.UpdateMessage.from_iu(new_iu, ut)
                else:
                    output_msg.add_iu(new_iu, ut)

        now = time.monotonic()
        if self._stat_n and now >= self._stat_next_report:
            self._stat_next_report = now + 2.0
            audio_s = self._stat_n / (self._session_rate or 16000)
            rms_in = (self._stat_in_sq / self._stat_n) ** 0.5
            rms_out = (self._stat_out_sq / self._stat_n) ** 0.5
            rt = audio_s / self._stat_proc_s if self._stat_proc_s > 0 else float("inf")
            gated = (f" gated={100 * self._stat_gated / self._stat_frames:.0f}%"
                     if self.gate_rms > 0 and self._stat_frames else "")
            print(f"[hush] {audio_s:.1f}s processed ({rt:.0f}x RT) "
                  f"rms in={rms_in:.4f} out={rms_out:.4f}{gated}")
            self._stat_in_sq = self._stat_out_sq = 0.0
            self._stat_n = 0
            self._stat_proc_s = 0.0
            self._stat_frames = 0
            self._stat_gated = 0

        if output_msg:
            self.append(output_msg)

    def shutdown(self):
        if self._session is not None:
            self._session.close()
            self._session = None
        super().shutdown()
