#!/usr/bin/env python3
"""
SYJ Voice Assistant - local voice orchestrator for Termux (Android)

Pipeline:
    Listen (Termux:API mic)  ->  Transcribe (whisper.cpp)
    ->  Process (Ollama / Llama 3)  ->  Speak (Piper TTS)

Requirements (install before running):
    pkg install termux-api ffmpeg          # + install the Termux:API app from F-Droid/Play Store
    whisper.cpp built locally (main / whisper-cli binary + a ggml model)
    ollama running locally with `ollama pull llama3`
    piper binary + a .onnx voice model

Note on "listen": Termux does not expose a raw ALSA/PulseAudio mic device,
so true continuous silence-detection (VAD) isn't reliable out of the box.
This script uses push-to-talk instead: press Enter to start recording,
press Enter again to stop. It's the most robust option on stock Termux.
If you later add a VAD library, swap out the `listen()` function only.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# CONFIG - edit these paths for your setup
# ---------------------------------------------------------------------------

# whisper.cpp: newer builds renamed the binary from `main` to `whisper-cli`.
# Point this at whichever one you built.
WHISPER_BIN = os.path.expanduser("~/whisper.cpp/main")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin")
WHISPER_LANG = "en"

PIPER_BIN = shutil.which("piper") or os.path.expanduser("~/piper/piper")
PIPER_MODEL = os.path.expanduser("~/piper/en_US-lessac-medium.onnx")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise personal assistant for Syed Ali Hasan. "
    "You are efficient and always ready to help."
)

MAX_RECORD_SECONDS = 30  # hard safety cap; recording still stops early on Enter

# ---------------------------------------------------------------------------
# Working files
# ---------------------------------------------------------------------------

WORKDIR = tempfile.mkdtemp(prefix="syj_assistant_")
CAPTURED_AUDIO = os.path.join(WORKDIR, "input.m4a")   # raw mic capture (aac container)
WHISPER_INPUT = os.path.join(WORKDIR, "input_16k.wav")  # 16kHz mono wav for whisper.cpp
WHISPER_OUT_PREFIX = os.path.join(WORKDIR, "input")
TRANSCRIPT_PATH = WHISPER_OUT_PREFIX + ".txt"
REPLY_WAV = os.path.join(WORKDIR, "reply.wav")


def check_dependencies():
    missing = []
    if not shutil.which("termux-microphone-record"):
        missing.append(
            "termux-microphone-record not found. Run: pkg install termux-api "
            "(and install the Termux:API companion app)"
        )
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg not found. Run: pkg install ffmpeg")
    if not os.path.isfile(WHISPER_BIN):
        missing.append(f"whisper.cpp binary not found at {WHISPER_BIN}")
    if not os.path.isfile(WHISPER_MODEL):
        missing.append(f"whisper.cpp model not found at {WHISPER_MODEL}")
    if not PIPER_BIN or not os.path.isfile(PIPER_BIN):
        missing.append("piper binary not found (check PIPER_BIN)")
    if not os.path.isfile(PIPER_MODEL):
        missing.append(f"piper voice model not found at {PIPER_MODEL}")
    if not (shutil.which("ffplay") or shutil.which("termux-media-player")):
        missing.append("no audio player found (install ffmpeg for ffplay, or termux-api)")

    if missing:
        print("Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 1. LISTEN
# ---------------------------------------------------------------------------

def listen() -> bool:
    """Push-to-talk capture via Termux:API. Returns True if audio was captured."""
    input("\nPress Enter to start speaking...")
    print("Listening... press Enter again to stop.")

    if os.path.exists(CAPTURED_AUDIO):
        os.remove(CAPTURED_AUDIO)

    try:
        subprocess.run(
            ["termux-microphone-record", "-f", CAPTURED_AUDIO, "-l", str(MAX_RECORD_SECONDS)],
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as e:
        print(f"[listen] Could not start recording: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("[listen] Recording command did not respond.")
        return False

    input()  # block here until the user presses Enter again

    try:
        subprocess.run(["termux-microphone-record", "-q"], check=True, timeout=10)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[listen] Could not stop recording cleanly: {e}")

    if not os.path.exists(CAPTURED_AUDIO) or os.path.getsize(CAPTURED_AUDIO) < 1000:
        print("[listen] No audio captured, try again.")
        return False
    return True


def convert_audio() -> bool:
    """Convert the raw capture to 16kHz mono WAV, which whisper.cpp expects."""
    if os.path.exists(WHISPER_INPUT):
        os.remove(WHISPER_INPUT)

    cmd = ["ffmpeg", "-y", "-i", CAPTURED_AUDIO, "-ar", "16000", "-ac", "1", WHISPER_INPUT]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        print("[convert_audio] ffmpeg not found.")
        return False
    except subprocess.TimeoutExpired:
        print("[convert_audio] ffmpeg timed out.")
        return False

    if result.returncode != 0 or not os.path.exists(WHISPER_INPUT):
        print(f"[convert_audio] ffmpeg failed:\n{result.stderr[-500:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# 2. TRANSCRIBE
# ---------------------------------------------------------------------------

def transcribe() -> str | None:
    if os.path.exists(TRANSCRIPT_PATH):
        os.remove(TRANSCRIPT_PATH)

    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-f", WHISPER_INPUT,
        "-otxt",
        "-of", WHISPER_OUT_PREFIX,
        "-nt",
        "-l", WHISPER_LANG,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        print(f"[transcribe] whisper.cpp binary not found at {WHISPER_BIN}")
        return None
    except subprocess.TimeoutExpired:
        print("[transcribe] whisper.cpp timed out.")
        return None

    if result.returncode != 0:
        print(f"[transcribe] whisper.cpp failed:\n{result.stderr[-500:]}")
        return None

    if not os.path.exists(TRANSCRIPT_PATH):
        print("[transcribe] No transcript produced.")
        return None

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("[transcribe] Empty transcript, ignoring.")
        return None

    print(f"You said: {text}")
    return text


# ---------------------------------------------------------------------------
# 3. PROCESS (Ollama)
# ---------------------------------------------------------------------------

def process(prompt_text: str) -> str | None:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt_text,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"[process] Could not reach Ollama at {OLLAMA_URL}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"[process] Bad JSON from Ollama: {e}")
        return None

    reply = body.get("response", "").strip()
    if not reply:
        print("[process] Ollama returned an empty response.")
        return None

    print(f"Assistant: {reply}")
    return reply


# ---------------------------------------------------------------------------
# 4. SPEAK (Piper)
# ---------------------------------------------------------------------------

def speak(text: str):
    if os.path.exists(REPLY_WAV):
        os.remove(REPLY_WAV)

    cmd = [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", REPLY_WAV]
    try:
        result = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        print(f"[speak] piper binary not found at {PIPER_BIN}")
        return
    except subprocess.TimeoutExpired:
        print("[speak] piper timed out.")
        return

    if result.returncode != 0 or not os.path.exists(REPLY_WAV):
        print(f"[speak] piper failed:\n{result.stderr[-500:]}")
        return

    play(REPLY_WAV)


def play(wav_path: str):
    # ffplay blocks until playback finishes, which keeps the loop in sync.
    # termux-media-player is used as a fallback but returns immediately
    # (it hands playback to Android's media session in the background).
    try:
        if shutil.which("ffplay"):
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", wav_path],
                check=True, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif shutil.which("termux-media-player"):
            subprocess.run(["termux-media-player", "play", wav_path], check=True, timeout=10)
        else:
            print("[play] No audio player available.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"[play] Could not play audio: {e}")


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def cleanup(*_):
    print("\nShutting down SYJ Voice Assistant...")
    shutil.rmtree(WORKDIR, ignore_errors=True)
    sys.exit(0)


def main():
    check_dependencies()
    signal.signal(signal.SIGINT, cleanup)

    print("SYJ Voice Assistant ready. Ctrl+C to quit.")
    while True:
        try:
            if not listen():
                continue
            if not convert_audio():
                continue
            text = transcribe()
            if not text:
                continue
            reply = process(text)
            if not reply:
                continue
            speak(reply)
        except Exception as e:
            # Catch-all so one bad turn never kills the whole session.
            print(f"[main loop] Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()
