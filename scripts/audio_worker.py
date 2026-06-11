#!/usr/bin/env python3
# audio_worker.py — optional centralized logging
import time
from pathlib import Path
import subprocess
from shutil import which  # ✅ minimal addition (needed to detect sox/ffmpeg)

from tunables import VALID_MIC_IDS, MIC_CHANNELS
from logger_utils import DebouncedLogger


def _split_stereo_to_two_mono(wav_final: Path):
    """
    Split stereo WAV into two mono WAVs:
      *_tx1.wav and *_tx2.wav
    Tries sox first, then ffmpeg.
    """
    base = wav_final.with_suffix("")  # path without .wav
    tx1 = Path(str(base) + "_tx1.wav")
    tx2 = Path(str(base) + "_tx2.wav")

    if which("sox"):
        subprocess.run(["sox", str(wav_final), str(tx1), "remix", "1"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["sox", str(wav_final), str(tx2), "remix", "2"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return

    if which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_final), "-map_channel", "0.0.0", str(tx1)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_final), "-map_channel", "0.0.1", str(tx2)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return


def audio_worker(device_str: str, out_dir: Path, duration_sec: float, rate: int, logger: DebouncedLogger | None = None):
    """
    Record from a single whitelisted ALSA device (e.g., 'hw:0,0') for duration_sec seconds.
    No auto-detection or fallback. If not in whitelist, skip.
    """
    # Logging is optional; if not provided, be quiet except for final success line.
    def _log_info(msg: str):
        if logger: logger.info(msg)
    def _log_warn(msg: str):
        if logger: logger.warn(msg)
    def _flush(force: bool = False):
        if logger: logger.periodic_flush(force=force)

    if device_str not in VALID_MIC_IDS:
        _log_warn(f"audio_worker: '{device_str}' not in whitelist → skipping.")
        _flush()
        return
    if duration_sec <= 0:
        _log_warn("audio_worker: duration_sec <= 0 → skipping.")
        _flush()
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Minimal fix: add milliseconds so multiple mics started in same second don't collide
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1000):03d}"

    # ✅ Minimal fix: sanitize '=' as well (CARD=RX,DEV=0)
    safe_name = device_str.replace(":", "").replace(",", "").replace("=", "")

    wav_tmp = out_dir / f"mic_{safe_name}_{ts}.part"
    wav_final = Path(str(wav_tmp).replace(".part", ".wav"))

    # ✅ Minimal update:
    # - Default is still mono (1 channel)
    # - For RODE Wireless GO II RX, MIC_CHANNELS maps this device to 2 channels
    channels = int(MIC_CHANNELS.get(device_str, 1))

    cmd = ["arecord", "-D", device_str, "-f", "cd", "-c", str(channels), "-r", str(rate), "-t", "wav", str(wav_tmp)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log_info(f"Audio started on {device_str} → {wav_tmp.name}")
    except Exception as e:
        _log_warn(f"audio({device_str}): failed to start arecord: {e}")
        _flush()
        return

    start = time.time()
    try:
        while (time.time() - start) < duration_sec:
            time.sleep(0.05)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        if wav_tmp.exists():
            wav_tmp.replace(wav_final)
            print(f"[✅] Audio saved: {wav_final}")  # keep one visible success line
            _log_info(f"Audio saved: {wav_final.name}")

            # ✅ Minimal addition:
            # If device is 2-channel (e.g., RODE Wireless GO II RX), split to two mono WAVs (TX1/TX2)
            if channels == 2:
                _split_stereo_to_two_mono(wav_final)

        else:
            _log_warn(f"audio({device_str}): no output file produced.")
    except Exception as e:
        _log_warn(f"audio({device_str}): {e}")
        try:
            proc.kill()
        except Exception:
            pass
    finally:
        _flush(force=True)
