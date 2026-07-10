"""
remote_duplex.py — drop-in retico module that talks to a REMOTE full-duplex
model over the WebSocket protocol instead of running the model locally.

Use it exactly like the local module::

    # before (runs the whole model on this machine):
    #   from retico_minicpm.minicpmduplex import MiniCPMDuplexModule
    #   duplex = MiniCPMDuplexModule()

    from remote_duplex import MiniCPMDuplexModule
    duplex = MiniCPMDuplexModule(server_url="https://f201fd249e6eb3f56e.gradio.live")

Nothing else in your retico graph changes: it consumes ``AudioIU`` (mic / filter
output) and produces ``AudioIU`` (for the speaker), just like the local module.

Under the hood it uses ``FullDuplexClient`` (duplex_client.py) which connects to
``server.py``'s ``/ws`` endpoint. A bare host or ``https://host`` URL is
normalised to ``wss://host/ws`` by ``server_url_from_address`` — so the
``*.gradio.live`` --share tunnel of the server works directly.

Requires only the lightweight client deps (``websockets``, ``numpy``,
``soundfile``); it does NOT import torch / the model / full_duplex.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import retico_core
import soundfile as sf
from retico_core import abstract

from duplex_client import FullDuplexClient

# Default to the share tunnel the user provided; override per-instance.
DEFAULT_SERVER_URL = "https://f201fd249e6eb3f56e.gradio.live"


class MiniCPMDuplexModule(abstract.AbstractModule):
    """Retico module backed by a remote full-duplex server (same public API as
    the local ``MiniCPMDuplexModule``)."""

    @staticmethod
    def name():
        return "RemoteMiniCPMDuplex"

    @staticmethod
    def description():
        return "Full-duplex model accessed over WebSocket (remote server)."

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        poll_interval_s: float = 0.02,
        client_name: str = "retico-remote",
        open_timeout: float = 10.0,
        record_wav: bool = True,
        wav_path: str = "remote_duplex_conversation.wav",
        wav_sample_rate: int = 16000,
        wav_flush_interval_s: float = 5.0,
        debug_audio: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.server_url = server_url
        self.poll_interval_s = poll_interval_s
        self.client_name = client_name
        self.open_timeout = open_timeout

        self.debug_audio = debug_audio
        self._fmt_logged = False
        self._dbg_count = 0

        self.client: Optional[FullDuplexClient] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False

        # --- conversation recording (debug) -------------------------------
        # Stereo WAV: left = user audio sent, right = bot audio received.
        # Each entry is (arrival_offset_s, sample_rate, float32_mono).
        self.record_wav = record_wav
        self.wav_path = wav_path
        self.wav_sample_rate = wav_sample_rate
        self.wav_flush_interval_s = wav_flush_interval_s
        self._rec_lock = threading.Lock()
        self._user_chunks: List[Tuple[float, int, np.ndarray]] = []
        self._bot_chunks: List[Tuple[float, int, np.ndarray]] = []
        self._rec_t0: Optional[float] = None
        self._last_flush = 0.0

    # ------------------------------------------------------------------ #
    # Incoming mic/filter audio  ->  remote server
    # ------------------------------------------------------------------ #
    def process_update(self, update_message):
        if self.client is None or not self.client.connected:
            return None

        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            # int16 PCM bytes -> float32 in [-1, 1], matching the local module.
            audio = np.frombuffer(iu.raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            if self.debug_audio:
                self._debug_chunk(iu, audio)

            self._record("user", iu.rate, audio)
            try:
                self.client.send_audio_chunk(iu.rate, audio)
            except Exception as exc:  # connection dropped mid-send
                print(f"[RemoteDuplex] send failed: {exc}")
        return None

    def _debug_chunk(self, iu, audio: np.ndarray):
        """One-shot format dump + periodic RMS so you can confirm IU decoding
        and see whether non-silent audio is actually leaving."""
        if not self._fmt_logged:
            self._fmt_logged = True
            print(
                "[RemoteDuplex] first mic IU: "
                f"rate={getattr(iu, 'rate', '?')} "
                f"sample_width={getattr(iu, 'sample_width', '?')} "
                f"nframes={getattr(iu, 'nframes', '?')} "
                f"raw_bytes={len(iu.raw_audio)} "
                f"-> int16 samples={audio.size} "
                f"(={audio.size / max(1, iu.rate):.3f}s)"
            )
        self._dbg_count += 1
        if self._dbg_count % 25 == 0:  # ~0.5s at 20ms frames
            rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            print(f"[RemoteDuplex] mic rms={rms:.4f} peak={peak:.3f} "
                  f"{'(SILENT)' if peak < 1e-3 else ''}")

    # ------------------------------------------------------------------ #
    # Remote TTS audio  ->  outgoing AudioIU (for the speaker)
    # ------------------------------------------------------------------ #
    def _make_tts_update_message(self, tts_chunk):
        sr, audio = tts_chunk  # audio is float32 in [-1, 1] from the client
        self._record("bot", sr, np.asarray(audio, dtype=np.float32))
        pcm16 = (np.asarray(audio, dtype=np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)

        output_iu = self.create_iu()
        output_iu.raw_audio = pcm16.tobytes()
        output_iu.rate = sr
        output_iu.nframes = len(pcm16)
        output_iu.sample_width = 2
        return abstract.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def _poll_loop(self):
        while self._running:
            client = self.client
            if client is None:
                time.sleep(self.poll_interval_s)
                continue
            chunk = client.pop_audio_chunk(timeout=self.poll_interval_s)
            if chunk is not None:
                self.append(self._make_tts_update_message(chunk))
            # surface any server-side warnings/errors
            for w in client.drain_warnings():
                print(f"[RemoteDuplex] {w.get('type')}: {w.get('message')}")
            # periodically persist the conversation WAV (survives hard kills)
            if self.record_wav and self.wav_flush_interval_s > 0:
                now = time.monotonic()
                if now - self._last_flush >= self.wav_flush_interval_s:
                    self._last_flush = now
                    self._write_wav()

    # ------------------------------------------------------------------ #
    # Conversation recording (debug)
    # ------------------------------------------------------------------ #
    def _record(self, channel: str, sample_rate: int, audio: np.ndarray):
        if not self.record_wav or audio.size == 0:
            return
        with self._rec_lock:
            if self._rec_t0 is None:
                self._rec_t0 = time.monotonic()
            offset = time.monotonic() - self._rec_t0
            bucket = self._user_chunks if channel == "user" else self._bot_chunks
            bucket.append((offset, int(sample_rate), audio.reshape(-1).copy()))

    @staticmethod
    def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out or audio.size == 0:
            return audio
        n_out = max(1, int(round(audio.size * sr_out / sr_in)))
        xp = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x, xp, audio).astype(np.float32)

    def _build_channel(self, chunks: List[Tuple[float, int, np.ndarray]]) -> np.ndarray:
        """Place each chunk at its arrival offset, never earlier than the end
        of the previous chunk. Bursty bot delivery stays back-to-back (a burst
        of TTS chunks arriving faster than realtime doesn't overlap), but the
        silent gaps between turns are PRESERVED — the old version dropped
        them, which slid every later bot utterance earlier in the file until
        answers appeared before the user's questions."""
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        sr = self.wav_sample_rate
        parts: List[np.ndarray] = []
        cursor = 0  # write position, in samples
        for offset, csr, a in chunks:
            start = int(offset * sr)
            if start > cursor:
                parts.append(np.zeros(start - cursor, dtype=np.float32))
                cursor = start
            body = self._resample(a, csr, sr)
            parts.append(body)
            cursor += len(body)
        return np.concatenate(parts)

    def _write_wav(self):
        with self._rec_lock:
            user = self._build_channel(list(self._user_chunks))
            bot = self._build_channel(list(self._bot_chunks))
        if user.size == 0 and bot.size == 0:
            return
        n = max(user.size, bot.size)
        left = np.zeros(n, dtype=np.float32)
        right = np.zeros(n, dtype=np.float32)
        left[: user.size] = user
        right[: bot.size] = bot
        stereo = np.stack([left, right], axis=1).clip(-1.0, 1.0)
        try:
            sf.write(self.wav_path, stereo, self.wav_sample_rate, subtype="PCM_16")
        except Exception as exc:
            print(f"[RemoteDuplex] WAV write failed: {exc}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def prepare_run(self):
        self.client = FullDuplexClient(self.server_url, open_timeout=self.open_timeout)
        session_id = self.client.connect(client_name=self.client_name)
        print(f"[RemoteDuplex] connected to {self.client.server_url} (session={session_id})")

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def shutdown(self):
        self._running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.record_wav:
            self._write_wav()
            print(f"[RemoteDuplex] conversation saved to {self.wav_path} "
                  f"(L=user, R=bot)")
        super().shutdown()
