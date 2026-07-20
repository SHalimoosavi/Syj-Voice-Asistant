#!/usr/bin/env python3
"""
SYJ Voice Assistant v2 - fully local voice orchestrator for Termux (Android)

Pipeline:
    Listen (Termux:API mic) -> Transcribe (whisper.cpp / whisper-cli)
    -> Process (llama.cpp server, local) -> Speak (Piper TTS)

No cloud calls, no Ollama. Every step is either a local binary or a
localhost HTTP call to a model server you run yourself.

--------------------------------------------------------------------------
ARCHITECTURE NOTE: why llama-server instead of calling a CLI binary
--------------------------------------------------------------------------
llama.cpp ships two relevant binaries: `llama-cli` (loads the model fresh
on every invocation) and `llama-server` (loads the model once, then answers
requests over HTTP). Reloading a multi-GB GGUF model on every turn would
make each reply take minutes on phone hardware, so this script assumes
llama-server is already running in the background and just talks to it
over HTTP - the same pattern v1 used with Ollama.

You need to start it yourself, in a separate Termux session, e.g.:
    llama-server -m ~/llama.cpp/models/<your-model>.gguf -c 4096 --port 8080
Keep that session alive (a second Termux tab, or `tmux`/`screen` +
`termux-wake-lock` so Android doesn't kill it in the background).

--------------------------------------------------------------------------
ARCHITECTURE NOTE: why push-to-talk instead of silence detection
--------------------------------------------------------------------------
Termux has no raw ALSA/PulseAudio mic device, so amplitude-based VAD isn't
reliable here. Press Enter to start recording, Enter again to stop - this
is a deliberate design choice carried over from v1, not a shortcut.

--------------------------------------------------------------------------
BEFORE RUNNING - confirm CLI flags for your exact builds
--------------------------------------------------------------------------
whisper-cli and llama-server flags occasionally change between versions.
If either step fails with an "unrecognized option" error, run:
    ~/whisper.cpp/build/bin/whisper-cli --help
    llama-server --help
and adjust the flags in transcribe() / the llama-server startup command
accordingly - don't assume this script's flags are permanently correct.
"""

import atexit
import glob
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
# CONFIG - persona and network endpoints
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise personal assistant for Syed Ali Hasan. "
    "You are efficient and always ready to help."
)

LLAMA_SERVER_URL = "http://localhost:8080"
MAX_RECORD_SECONDS = 30  # safety cap; recording still stops early on Enter

# ---------------------------------------------------------------------------
# AUTODETECTION - binaries and models
# ---------------------------------------------------------------------------

def first_existing(paths):
    """Return the first path in the list that actually exists on disk."""
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def find_whisper_bin():
    candidates = [
        os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli"),  # your CMake build
        os.path.expanduser("~/whisper.cpp/main"),                    # older make-based build
        shutil.which("whisper-cli"),
    ]
    return first_existing(candidates)


def find_whisper_model():
    model_dir = os.path.expanduser("~/whisper.cpp/models")
    preferred = os.path.join(model_dir, "ggml-base.en.bin")
    if os.path.isfile(preferred):
        return preferred
    matches = sorted(glob.glob(os.path.join(model_dir, "ggml-*.bin")))
    return matches[0] if matches else None


def find_piper_bin():
    # The apt-installed gyroing/piper-tts-for-termux .deb puts this on PATH.
    candidates = [
        shutil.which("piper"),
        os.path.expanduser("~/piper/piper"),
        os.path.expanduser("~/piper-tts/piper"),
    ]
    return first_existing(candidates)


def find_piper_model():
    search_dirs = [os.path.expanduser("~/piper"), os.path.expanduser("~/piper-tts")]
    for d in search_dirs:
        matches = sorted(glob.glob(os.path.join(d, "*.onnx")))
        if matches:
            return matches[0]
    return None


WHISPER_BIN = find_whisper_bin()
WHISPER_MODEL = find_whisper_model()
PIPER_BIN = find_piper_bin()
PIPER_MODEL = find_piper_model()
# The Termux Piper CLI takes a bare voice name (no .onnx extension) resolved
# via the PIPER_VOICE_PATH env var, not a full file path.
PIPER_VOICE_DIR = os.path.dirname(PIPER_MODEL) if PIPER_MODEL else None
PIPER_VOICE_NAME = os.path.splitext(os.path.basename(PIPER_MODEL))[0] if PIPER_MODEL else None
WHISPER_LANG = "en"

# ---------------------------------------------------------------------------
# Working files
# ---------------------------------------------------------------------------

WORKDIR = tempfile.mkdtemp(prefix="syj_assistant_v2_")
atexit.register(lambda: shutil.rmtree(WORKDIR, ignore_errors=True))
CAPTURED_AUDIO = os.path.join(WORKDIR, "input.m4a")
WHISPER_INPUT = os.path.join(WORKDIR, "input_16k.wav")
WHISPER_OUT_PREFIX = os.path.join(WORKDIR, "input")
TRANSCRIPT_PATH = WHISPER_OUT_PREFIX + ".txt"
REPLY_WAV = os.path.join(WORKDIR, "reply.wav")

# Persistent (not auto-deleted) copies of the last turn's audio, so you can
# manually play them back to check what was actually captured/trimmed -
# useful when the transcript looks wrong and you need to know which stage
# lost the audio.
DEBUG_DIR = os.path.expanduser("~/syj_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
DEBUG_RAW = os.path.join(DEBUG_DIR, "last_raw.m4a")
DEBUG_TRIMMED = os.path.join(DEBUG_DIR, "last_trimmed.wav")


def get_duration_seconds(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STATUS / DEPENDENCY CHECK
# ---------------------------------------------------------------------------

def check_llama_server_reachable() -> bool:
    try:
        req = urllib.request.Request(f"{LLAMA_SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


def print_status_and_exit_if_blocked():
    rows = [
        ("Termux:API (mic)", bool(shutil.which("termux-microphone-record"))),
        ("FFmpeg", bool(shutil.which("ffmpeg"))),
        ("whisper-cli binary", bool(WHISPER_BIN)),
        ("Whisper model", bool(WHISPER_MODEL)),
        ("Piper binary", bool(PIPER_BIN)),
        ("Piper voice model", bool(PIPER_MODEL)),
        ("Audio player", bool(shutil.which("termux-media-player") or shutil.which("ffplay"))),
        ("llama-server reachable", check_llama_server_reachable()),
    ]

    print("Component status")
    blocking = False
    for name, ok in rows:
        print(f"  {'OK ' if ok else 'MISSING'}  {name}")
        if not ok:
            blocking = True

    if not blocking:
        print("\nAll components ready.\n")
        return

    print("\nFix the missing pieces above before running:")
    if not shutil.which("termux-microphone-record"):
        print("  - pkg install termux-api  (and install the Termux:API app)")
    if not shutil.which("ffmpeg"):
        print("  - pkg install ffmpeg")
    if not WHISPER_BIN:
        print("  - whisper-cli not found. Expected at ~/whisper.cpp/build/bin/whisper-cli")
    if not WHISPER_MODEL:
        print("  - No ggml model found in ~/whisper.cpp/models/")
    if not PIPER_BIN:
        print("  - Piper not installed. Stock Piper releases are glibc-only and won't run")
        print("    on Termux's Bionic libc. Use the Termux-native build instead:")
        print("    pkg install espeak")
        print("    Download the .deb from:")
        print("    https://github.com/gyroing/piper-tts-for-termux/releases/tag/v1.2-android-termux")
        print("    Then: apt install -f ~/<downloaded-file>.deb")
    if not PIPER_MODEL:
        print("  - No Piper voice model (.onnx) found in ~/piper/. Example (Lessac, US English):")
        print("    wget -P ~/piper https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
        print("    wget -P ~/piper https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json")
    if not (shutil.which("termux-media-player") or shutil.which("ffplay")):
        print("  - No audio player found. termux-media-player comes with Termux:API")
        print("    (already required above); ffplay is a fallback but needs PulseAudio.")
    if not check_llama_server_reachable():
        model_hint = "~/llama.cpp/models/<your-model>.gguf"
        print(f"  - llama-server not reachable at {LLAMA_SERVER_URL}. Start it in another")
        print(f"    Termux session: llama-server -m {model_hint} -c 4096 --port 8080")

    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. LISTEN (unchanged from v1 - already validated on your device)
# ---------------------------------------------------------------------------

def listen() -> bool:
    input("\nPress Enter to start speaking...")
    print("Listening... press Enter again to stop.")

    if os.path.exists(CAPTURED_AUDIO):
        os.remove(CAPTURED_AUDIO)

    try:
        result = subprocess.run(
            ["termux-microphone-record", "-f", CAPTURED_AUDIO, "-l", str(MAX_RECORD_SECONDS)],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        print("[listen] Recording command did not respond.")
        return False

    # termux-microphone-record often returns exit code 0 even when it fails -
    # the failure shows up as a JSON error in stdout instead. Check for it
    # explicitly rather than trusting the return code.
    output = (result.stdout or "").strip()
    if output:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and parsed.get("error"):
                print(f"[listen] Termux:API error: {parsed['error']}")
                print("Fix: Android Settings > Apps > Termux:API > Permissions > Microphone > Allow")
                return False
        except json.JSONDecodeError:
            pass  # non-JSON stdout on a successful start is normal

    input()  # blocks until Enter is pressed again

    try:
        stop_result = subprocess.run(
            ["termux-microphone-record", "-q"], capture_output=True, text=True, timeout=10
        )
        stop_output = (stop_result.stdout or "").strip()
        if stop_output:
            try:
                parsed = json.loads(stop_output)
                if isinstance(parsed, dict) and parsed.get("error"):
                    print(f"[listen] Termux:API error while stopping: {parsed['error']}")
            except json.JSONDecodeError:
                pass
    except subprocess.TimeoutExpired:
        print("[listen] Could not stop recording cleanly (timed out).")

    if not os.path.exists(CAPTURED_AUDIO) or os.path.getsize(CAPTURED_AUDIO) < 1000:
        print("[listen] No audio captured, try again.")
        return False

    shutil.copy2(CAPTURED_AUDIO, DEBUG_RAW)
    dur = get_duration_seconds(CAPTURED_AUDIO)
    if dur is not None:
        print(f"[listen] Captured {dur:.1f}s of raw audio -> {DEBUG_RAW}")
        if dur < 0.6:
            print("[listen] That's very short - the mic may have started late. "
                  "Try pausing briefly after pressing Enter before you speak.")
    return True


def convert_audio() -> bool:
    if os.path.exists(WHISPER_INPUT):
        os.remove(WHISPER_INPUT)

    # Two-pass reverse-based trim: this only strips silence at the very
    # start and very end of the clip. The single-pass version (start+stop
    # combined in one filter call) was mistaking natural pauses BETWEEN
    # WORDS for the final trailing silence and truncating mid-sentence -
    # confirmed against a real recording that got cut from 11s down to
    # 0.8s, wiping out an entire spoken question.
    cmd = [
        "ffmpeg", "-y", "-i", CAPTURED_AUDIO,
        "-af", "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB,"
               "areverse,"
               "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB,"
               "areverse",
        "-ar", "16000", "-ac", "1", WHISPER_INPUT,
    ]
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

    shutil.copy2(WHISPER_INPUT, DEBUG_TRIMMED)
    raw_dur = get_duration_seconds(CAPTURED_AUDIO)
    trimmed_dur = get_duration_seconds(WHISPER_INPUT)
    if raw_dur is not None and trimmed_dur is not None:
        print(f"[convert_audio] After silence trim: {trimmed_dur:.1f}s (raw was {raw_dur:.1f}s) -> {DEBUG_TRIMMED}")
        if trimmed_dur < 0.3:
            print("[convert_audio] Almost everything got trimmed as \"silence\". Play "
                  f"{DEBUG_RAW} with termux-media-player to check what was actually recorded.")
        elif raw_dur > 2.0 and trimmed_dur < raw_dur * 0.3:
            print("[convert_audio] More than 70% of the recording got trimmed - if the "
                  "transcript below looks wrong or short, this is likely why. Play "
                  f"{DEBUG_RAW} vs {DEBUG_TRIMMED} to compare.")
    return True


# ---------------------------------------------------------------------------
# 2. TRANSCRIBE (whisper-cli, path fixed for your CMake build)
# ---------------------------------------------------------------------------

def transcribe():
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
        print(f"[transcribe] whisper-cli not found at {WHISPER_BIN}")
        return None
    except subprocess.TimeoutExpired:
        print("[transcribe] whisper-cli timed out.")
        return None

    if result.returncode != 0:
        print(f"[transcribe] whisper-cli failed:\n{result.stderr[-500:]}")
        print("If this mentions an unrecognized flag, run "
              f"'{WHISPER_BIN} --help' and compare against transcribe() in this script.")
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
# 3. PROCESS (llama.cpp server, OpenAI-compatible chat endpoint)
# ---------------------------------------------------------------------------

def process(prompt_text: str):
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0.7,
        "max_tokens": 300,  # keeps replies fast on phone hardware and concise
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"[process] Could not reach llama-server at {LLAMA_SERVER_URL}: {e}")
        print("Make sure llama-server is running in another Termux session.")
        return None
    except json.JSONDecodeError as e:
        print(f"[process] Bad JSON from llama-server: {e}")
        return None

    try:
        reply = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        print(f"[process] Unexpected response shape from llama-server: {body}")
        return None

    if not reply:
        print("[process] llama-server returned an empty response.")
        return None

    print(f"Assistant: {reply}")
    return reply


# ---------------------------------------------------------------------------
# 4. SPEAK (Piper - gyroing/piper-tts-for-termux build)
# ---------------------------------------------------------------------------
# This Termux-native fork's CLI differs from stock Piper:
#   -m VOICE_NAME    bare voice name, no .onnx extension, resolved via
#                     the PIPER_VOICE_PATH env var
#   -f OUTPUT_FILE   writes 32-bit float RAW PCM at 22050Hz mono
#                    (NOT a .wav file) - we convert it with ffmpeg below.

REPLY_RAW = os.path.join(WORKDIR, "reply.raw")


def speak(text: str):
    if os.path.exists(REPLY_RAW):
        os.remove(REPLY_RAW)
    if os.path.exists(REPLY_WAV):
        os.remove(REPLY_WAV)

    env = os.environ.copy()
    if PIPER_VOICE_DIR:
        env["PIPER_VOICE_PATH"] = PIPER_VOICE_DIR

    cmd = [PIPER_BIN, "-m", PIPER_VOICE_NAME, "-f", REPLY_RAW]
    try:
        result = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=60, env=env
        )
    except FileNotFoundError:
        print(f"[speak] piper binary not found at {PIPER_BIN}")
        return
    except subprocess.TimeoutExpired:
        print("[speak] piper timed out.")
        return

    if result.returncode != 0 or not os.path.exists(REPLY_RAW):
        print(f"[speak] piper failed:\n{result.stderr[-500:]}")
        print(f"If this mentions a flag or espeak-data error, run '{PIPER_BIN} -h'.")
        return

    convert_cmd = [
        "ffmpeg", "-y", "-f", "f32le", "-ar", "22050", "-ac", "1",
        "-i", REPLY_RAW, REPLY_WAV,
    ]
    try:
        conv = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        print("[speak] ffmpeg timed out converting Piper's raw output.")
        return

    if conv.returncode != 0 or not os.path.exists(REPLY_WAV):
        print(f"[speak] Could not convert Piper's raw output to WAV:\n{conv.stderr[-500:]}")
        return

    play(REPLY_WAV)


def play(wav_path: str):
    # termux-media-player (Android's native MediaPlayer via Termux:API) works
    # out of the box. ffplay needs a PulseAudio server, which stock Termux
    # doesn't have configured - so it's the fallback, not the default.
    tried = []

    if shutil.which("termux-media-player"):
        tried.append("termux-media-player")
        try:
            subprocess.run(["termux-media-player", "play", wav_path], check=True, timeout=10)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[play] termux-media-player failed: {e}")

    if shutil.which("ffplay"):
        tried.append("ffplay")
        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", wav_path],
                check=True, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[play] ffplay failed (needs PulseAudio: pkg install pulseaudio "
                  f"&& pulseaudio --start): {e}")

    if not tried:
        print("[play] No audio player available.")


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def cleanup(*_):
    print("\nShutting down SYJ Voice Assistant v2...")
    shutil.rmtree(WORKDIR, ignore_errors=True)
    sys.exit(0)


def main():
    print_status_and_exit_if_blocked()
    signal.signal(signal.SIGINT, cleanup)

    print("SYJ Voice Assistant v2 ready. Ctrl+C to quit.")
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
            print(f"[main loop] Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()
