"""edge-tts and Kokoro TTS generation for chronicle-forge."""
import asyncio
import os
import shutil
import subprocess
import tempfile

import edge_tts
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EDGE_VOICE   = os.getenv("TTS_VOICE", "en-US-AndrewMultilingualNeural")
DEFAULT_KOKORO_VOICE = "af_heart"


# ── edge-tts ──────────────────────────────────────────────────────────────────

async def _edge_stream(text: str, path: str, voice: str) -> list:
    communicate  = edge_tts.Communicate(text, voice)
    audio_chunks = []
    boundaries   = []

    async for event in communicate.stream():
        if event["type"] == "audio":
            audio_chunks.append(event["data"])
        elif event["type"] == "WordBoundary":
            # edge-tts returns offset/duration in 100-nanosecond units → ms
            boundaries.append({
                "word":        event["text"],
                "offset_ms":   event["offset"]   / 10_000,
                "duration_ms": event["duration"] / 10_000,
            })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)

    return boundaries


def generate_tts_edge(
    text: str,
    output_path: str = "output/narration.mp3",
    voice: str = None,
) -> tuple:
    """Generate TTS via edge-tts. Returns (output_path, word_boundaries)."""
    voice = voice or DEFAULT_EDGE_VOICE
    boundaries = asyncio.run(_edge_stream(text.strip(), output_path, voice))
    print(f"  TTS saved: {output_path}  ({len(boundaries)} word events)")
    return output_path, boundaries


# ── Kokoro TTS ────────────────────────────────────────────────────────────────

def _approximate_boundaries(text: str, duration_s: float) -> list:
    """Distribute word timestamps evenly across audio duration."""
    words = text.split()
    if not words or duration_s <= 0:
        return []
    ms_per_word = (duration_s * 1000) / len(words)
    return [
        {
            "word":        word,
            "offset_ms":   i * ms_per_word,
            "duration_ms": ms_per_word * 0.8,
        }
        for i, word in enumerate(words)
    ]


def generate_tts_kokoro(
    text: str,
    output_path: str = "output/narration.wav",
    voice: str = None,
    speed: float = 1.0,
) -> tuple:
    """
    Generate TTS via Kokoro (local, CPU-native).
    Returns (output_path, word_boundaries).

    Word boundaries are approximated by distributing timestamps evenly —
    Kokoro does not emit word-level events. For scenes with very short
    text_segments (<5 words), timing precision is reduced vs edge-tts.

    Requires: pip install kokoro soundfile
    """
    try:
        from kokoro import KPipeline
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "Kokoro not installed.\n"
            "Install with: pip install kokoro soundfile"
        )

    voice     = voice or DEFAULT_KOKORO_VOICE
    lang_code = "b" if voice.startswith("b") else "a"   # British vs American
    pipe      = KPipeline(lang_code=lang_code)

    print(f"  Kokoro: synthesising voice={voice}...")
    chunks = []
    for _, _, audio in pipe(text.strip(), voice=voice, speed=speed):
        chunks.append(audio)

    import numpy as np
    import soundfile as sf

    audio       = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sample_rate = 24_000
    duration_s  = len(audio) / sample_rate

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sf.write(output_path, audio, sample_rate)

    boundaries = _approximate_boundaries(text.strip(), duration_s)
    print(f"  TTS saved: {output_path}  ({duration_s:.1f}s, ~{len(boundaries)} words, "
          f"estimated timestamps)")
    return output_path, boundaries


# ── Unified entry point ───────────────────────────────────────────────────────

def generate_tts(
    text: str,
    output_path: str = None,
    voice: str = None,
    engine: str = "edge",
) -> tuple:
    """
    Generate TTS narration.  engine: 'edge' or 'kokoro'.
    Returns (output_path, [{word, offset_ms, duration_ms}]).
    """
    if engine == "kokoro":
        return generate_tts_kokoro(text, output_path or "output/narration.wav", voice)
    return generate_tts_edge(text, output_path or "output/narration.mp3", voice)


# ── Per-scene TTS (new architecture) ─────────────────────────────────────────

_INTER_REQUEST_DELAY = 0.25       # seconds between edge-tts calls (Azure throttle buffer)
_SCENE_TIMEOUT       = 30         # hard wall-clock limit per scene (asyncio.wait_for)
_RETRY_DELAYS        = [1.0, 3.0] # backoff between attempts 1→2 and 2→3


def _mp3_bytes_to_ms(clip_bytes: bytes) -> float:
    """
    Compute MP3 duration (ms) from byte length by parsing the first frame header.
    Falls back to 48 kbps — the bitrate edge-tts neural voices produce.

    The old fallback assumed 24 kbps which caused a 2× overestimate for edge-tts
    output at 48 kbps, making every scene's duration double its actual length.
    """
    _MPEG1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    _MPEG2_L3 = [0,  8, 16, 24, 32, 40, 48, 56,  64,  80,  96, 112, 128, 144, 160, 0]

    # Skip ID3v2 tag if present.
    offset = 0
    if len(clip_bytes) >= 10 and clip_bytes[:3] == b"ID3":
        sz = (clip_bytes[6] << 21 | clip_bytes[7] << 14 |
              clip_bytes[8] <<  7 | clip_bytes[9])
        offset = 10 + sz

    # Search for MPEG sync word within the first 2 KB after the tag.
    limit = min(len(clip_bytes) - 3, offset + 2048)
    for i in range(offset, limit):
        b0, b1 = clip_bytes[i], clip_bytes[i + 1]
        if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
            mpeg_ver = (b1 >> 3) & 0x03   # 3=MPEG1, 2=MPEG2/2.5
            layer    = (b1 >> 1) & 0x03   # 1=Layer3
            if layer != 0x01:
                continue
            br_idx  = (clip_bytes[i + 2] >> 4) & 0x0F
            bitrate = (_MPEG1_L3 if mpeg_ver == 3 else _MPEG2_L3)[br_idx]
            if bitrate > 0:
                # duration_ms = len_bytes × 8 bits/byte ÷ bitrate_kbps
                return max(300.0, len(clip_bytes) * 8.0 / bitrate)

    # Could not parse — assume 48 kbps (edge-tts neural voice default).
    return max(300.0, len(clip_bytes) / 6.0)


async def _edge_one_scene(text: str, voice: str) -> tuple:
    """
    Stream a single edge-tts segment with retries and a hard timeout.
    Returns (clip_bytes, clip_ms).  Never raises — returns (b"", 0.0) after all retries.
    """
    safe_text = text.strip()
    if len(safe_text) < 3:
        safe_text = (safe_text + ".") if safe_text else "..."

    last_err = ""
    for attempt in range(len(_RETRY_DELAYS) + 1):   # 3 attempts total
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            print(f"\n  TTS retry {attempt}/{len(_RETRY_DELAYS)} "
                  f"({last_err[:60]}) — waiting {delay:.0f}s...", flush=True)
            await asyncio.sleep(delay)

        audio_chunks: list = []
        boundaries:   list = []

        async def _do_stream():
            communicate = edge_tts.Communicate(
                safe_text, voice,
                connect_timeout=10,
                receive_timeout=20,
            )
            async for event in communicate.stream():
                if event["type"] == "audio":
                    audio_chunks.append(event["data"])
                elif event["type"] == "WordBoundary":
                    boundaries.append({
                        "offset_ms":   event["offset"]   / 10_000,
                        "duration_ms": event["duration"] / 10_000,
                    })

        try:
            await asyncio.wait_for(_do_stream(), timeout=_SCENE_TIMEOUT)

            clip_bytes = b"".join(audio_chunks)
            if not clip_bytes:
                raise ValueError("no audio bytes received")

            if boundaries:
                last    = boundaries[-1]
                clip_ms = last["offset_ms"] + last["duration_ms"] + 200
            else:
                clip_ms = _mp3_bytes_to_ms(clip_bytes)

            return clip_bytes, clip_ms

        except asyncio.TimeoutError:
            last_err = f"timeout after {_SCENE_TIMEOUT}s"
        except Exception as exc:
            last_err = str(exc)[:100]

    print(f"\n  Warning: TTS skipped for {safe_text[:50]!r} — {last_err}")
    return b"", 0.0


async def _edge_scenes_async(scenes: list, voice: str) -> tuple:
    """
    Stream one edge-tts clip per scene.
    Returns (clips_bytes, durations_ms) — one entry per scene.
    """
    clips     = []
    durations = []
    n         = len(scenes)

    for i, scene in enumerate(scenes):
        text = scene.get("text_segment", "").strip()
        if not text:
            clips.append(b"")
            durations.append(0.0)
            continue

        clip_bytes, clip_ms = await _edge_one_scene(text, voice)
        clips.append(clip_bytes)
        durations.append(clip_ms)

        if i < n - 1:
            await asyncio.sleep(_INTER_REQUEST_DELAY)

        print(f"\r  TTS scenes: {i+1}/{n} ({sum(durations)/1000:.1f}s total)",
              end="", flush=True)

    print()
    return clips, durations


def generate_tts_scenes(
    scenes: list,
    voice: str = None,
    engine: str = "edge",
) -> tuple:
    """
    Generate TTS with per-scene clips so start_ms/end_ms are exact.

    Each scene gets its own audio clip; clips are concatenated into one
    output file. scene["start_ms"] and scene["end_ms"] are set from the
    cumulative clip durations — no word-boundary matching needed.

    edge-tts : clips concatenated with FFmpeg → output/narration.mp3
    Kokoro   : clips concatenated as numpy arrays → output/narration.wav

    Returns (audio_path, scenes) with timestamps set on each scene dict.
    """
    if engine == "kokoro":
        return _generate_tts_scenes_kokoro(scenes, voice)
    return _generate_tts_scenes_edge(scenes, voice)


def _generate_tts_scenes_edge(scenes: list, voice: str = None) -> tuple:
    """edge-tts per-scene path. Concatenates clips via FFmpeg."""
    voice     = voice or DEFAULT_EDGE_VOICE
    n         = len(scenes)
    out_path  = "output/narration.mp3"

    clips, durations = asyncio.run(_edge_scenes_async(scenes, voice))

    # Assign timestamps from cumulative durations
    cursor_ms = 0.0
    for scene, clip_ms in zip(scenes, durations):
        scene["start_ms"] = int(cursor_ms)
        scene["end_ms"]   = int(cursor_ms + clip_ms)
        cursor_ms        += clip_ms

    total_ms = cursor_ms
    print(f"  {n} scene clips  |  total: {total_ms/1000:.1f}s")

    # Write temp clips and concat with FFmpeg
    tmp_dir  = tempfile.mkdtemp(prefix="cf_tts_")
    try:
        clip_paths = []
        for i, clip_bytes in enumerate(clips):
            if not clip_bytes:
                continue
            p = os.path.join(tmp_dir, f"clip_{i:04d}.mp3")
            with open(p, "wb") as f:
                f.write(clip_bytes)
            clip_paths.append(p)

        if not clip_paths:
            raise RuntimeError("No audio generated for any scene")

        list_file = os.path.join(tmp_dir, "concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        os.makedirs("output", exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", out_path],
            check=True, capture_output=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"  TTS saved: {out_path}")
    return out_path, scenes


def _generate_tts_scenes_kokoro(scenes: list, voice: str = None) -> tuple:
    """Kokoro per-scene path. Concatenates numpy arrays."""
    try:
        from kokoro import KPipeline
        import numpy as np
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "Kokoro not installed.\n"
            "Install with: pip install kokoro soundfile"
        )

    voice     = voice or DEFAULT_KOKORO_VOICE
    lang_code = "b" if voice.startswith("b") else "a"
    pipe      = KPipeline(lang_code=lang_code)

    import numpy as np
    import soundfile as sf

    SAMPLE_RATE   = 24_000
    out_path      = "output/narration.wav"
    n             = len(scenes)
    all_arrays    = []
    cursor_ms     = 0.0

    for i, scene in enumerate(scenes):
        text = scene.get("text_segment", "").strip()
        if not text:
            scene["start_ms"] = int(cursor_ms)
            scene["end_ms"]   = int(cursor_ms)
            all_arrays.append(np.zeros(0, dtype=np.float32))
            continue

        chunks = []
        for _, _, audio in pipe(text, voice=voice, speed=1.0):
            chunks.append(audio)

        clip_array = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        clip_ms    = len(clip_array) / SAMPLE_RATE * 1000

        scene["start_ms"] = int(cursor_ms)
        scene["end_ms"]   = int(cursor_ms + clip_ms)
        cursor_ms        += clip_ms
        all_arrays.append(clip_array)

        print(f"\r  TTS scenes: {i+1}/{n} ({cursor_ms/1000:.1f}s total)",
              end="", flush=True)

    print()

    full_audio = np.concatenate(all_arrays) if all_arrays else np.zeros(0, dtype=np.float32)
    os.makedirs("output", exist_ok=True)
    sf.write(out_path, full_audio, SAMPLE_RATE)
    print(f"  TTS saved: {out_path}  ({cursor_ms/1000:.1f}s, exact timestamps)")
    return out_path, scenes
