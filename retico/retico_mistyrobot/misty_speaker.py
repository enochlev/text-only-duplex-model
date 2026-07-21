import os
import wave
import tempfile

import requests

from retico_core import AbstractConsumingModule, UpdateType
from retico_core.audio import AudioIU


class MistySpeakerModule(AbstractConsumingModule):

    @staticmethod
    def name():
        return "Misty Speaker Module"

    @staticmethod
    def description():
        return "Uploads audio to Misty and plays it."

    @staticmethod
    def input_ius():
        return AudioIU

    @staticmethod
    def output_iu():
        return None

    def __init__(
        self,
        ip,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        volume=10,
        autoplay=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ip = ip
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.volume = volume
        self.autoplay = autoplay

        self.audio_buffer = bytearray()

        # Fixed ring of filenames reused with OverwriteExisting=True. A fresh
        # uuid per chunk accumulated forever on the robot — at ~2,470 files the
        # 820's asset service hangs on every list/save (10s IPC timeout) and
        # uploads start failing with "Could not find location of newly saved
        # audio file" (observed 2026-07-16, took both robots down). Ten names
        # is plenty: chunks are ~1s and play immediately, so a name is free
        # again long before the ring wraps.
        self._name_ring = [f"retico_chunk_{i}.wav" for i in range(10)]
        self._ring_idx = 0

    def process_update(self, update_message):
        if not update_message:
            return

        for iu, ut in update_message:
            if ut != UpdateType.ADD:
                continue

            # The WAV header must match the stream's ACTUAL rate (the duplex
            # server's TTS rate) or Misty plays it slowed down / sped up.
            rate = int(getattr(iu, "rate", 0) or 0)
            if rate and rate != self.sample_rate:
                print(f"[misty] adopting stream rate {rate} Hz "
                      f"(was configured {self.sample_rate})")
                self.sample_rate = rate

            if hasattr(iu, "payload") and iu.payload is not None:
                self.audio_buffer.extend(iu.payload)
            elif hasattr(iu, "raw_audio"):
                self.audio_buffer.extend(iu.raw_audio)
            elif hasattr(iu, "audio"):
                self.audio_buffer.extend(iu.audio)

            # flush roughly every second of audio
            if len(self.audio_buffer) >= self.sample_rate * self.sample_width:
                self._play_buffer()
                self.audio_buffer = bytearray()

    def shutdown(self):
        if len(self.audio_buffer) > 0:
            try:
                self._play_buffer()
            except Exception as exc:
                print(f"[misty] final flush failed: {exc}")
            self.audio_buffer = bytearray()
        super().shutdown()

    def _play_buffer(self):

        filename = self._name_ring[self._ring_idx]
        self._ring_idx = (self._ring_idx + 1) % len(self._name_ring)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(bytes(self.audio_buffer))

            self._upload_audio(filename, wav_path)

            if self.autoplay:
                self._play_audio(filename)

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def _upload_audio(self, filename, filepath):

        url = f"http://{self.ip}/api/audio"

        with open(filepath, "rb") as f:
            files = {
                "File": (filename, f, "audio/wav")
            }
            data = {
                "FileName": filename,
                "ImmediatelyApply": False,
                "OverwriteExisting": True,
            }
            response = requests.post(url, files=files, data=data, timeout=30)
            response.raise_for_status()

    def _play_audio(self, filename):

        url = f"http://{self.ip}/api/audio/play"

        payload = {
            "FileName": filename,
            "Volume": self.volume,
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
