"""Timestamp-synced Ken Burns video composer for chronicle-forge.

Key difference from story-reel's composer:
  Each scene has start_ms / end_ms derived from TTS word boundaries,
  so each image is shown for exactly as long as its narration is spoken.
  This is the tight TTS-image sync the pipeline requires.

Always outputs 1920×1080 (16:9).
"""
import os

import cv2
import numpy as np
from moviepy.editor import AudioFileClip, VideoClip, concatenate_videoclips

FPS         = 24
START_ZOOM  = 1.08   # subtle Ken Burns zoom-out (less aggressive than story-reel)
CROSSFADE_S = 0.25   # seconds of crossfade between scenes


def _fit_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale  = max(w / iw, h / ih)
    nw, nh = int(iw * scale) + 1, int(ih * scale) + 1
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    x = (nw - w) // 2
    y = (nh - h) // 2
    return resized[y:y + h, x:x + w]


def _kenburns_clip(img_path: str, duration: float, w: int, h: int) -> VideoClip:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {img_path}")
    base = _fit_crop(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        int(w * START_ZOOM) + 4,
        int(h * START_ZOOM) + 4,
    )
    big_h, big_w = base.shape[:2]

    def make_frame(t: float) -> np.ndarray:
        progress = min(1.0, t / max(duration - 1 / FPS, 1e-6))
        zoom  = START_ZOOM - (START_ZOOM - 1.0) * progress   # 1.08 → 1.0
        cw    = int(w * zoom)
        ch    = int(h * zoom)
        x     = (big_w - cw) // 2
        y     = (big_h - ch) // 2
        return cv2.resize(base[y:y + ch, x:x + cw], (w, h),
                          interpolation=cv2.INTER_LINEAR)

    return VideoClip(make_frame, duration=duration).set_fps(FPS)


def _find_word_sequence(wb_words: list, target_words: list, start_from: int = 0) -> int:
    """
    Find the first index >= start_from in wb_words where target_words appear in
    sequence.  Prefix-matches the first word as a fallback.
    The start_from constraint enforces monotone ordering across scenes.
    """
    if not target_words:
        return start_from
    first = target_words[0]
    for i in range(start_from, len(wb_words)):
        if wb_words[i] == first:
            if all(
                i + j < len(wb_words) and wb_words[i + j] == target_words[j]
                for j in range(len(target_words))
            ):
                return i
    # Prefix-match fallback for first word only
    for i in range(start_from, len(wb_words)):
        if len(first) >= 4 and wb_words[i].startswith(first[:4]):
            return i
        if wb_words[i] == first:
            return i
    return -1


def assign_scene_timestamps(
    scenes: list,
    word_boundaries: list,
    audio_duration_ms: float,
    narration: str = "",
) -> list:
    """
    Assign start_ms / end_ms to each scene.  Three strategies in priority order:

    1. Word-boundary monotone matching — searches forward from the previous
       match so repeated phrases can never map to the wrong occurrence.
       Uses up to 5 words for more discriminating matches.
    2. Character-position proportional — finds each text_segment in the
       narration string and maps its character offset to a time offset.
       Accurate when text_segments are exact substrings of narration.
    3. Even distribution — last resort when neither of the above applies.
    """
    n = max(len(scenes), 1)

    # ── Strategy 2 helper ────────────────────────────────────────────────────
    def _by_char_pos():
        norm_narr = narration.lower()
        total     = max(len(narration), 1)
        cursor    = 0
        for scene in scenes:
            seg      = scene.get("text_segment", "")
            norm_seg = seg.lower()
            # Match up to the first 50 characters for speed; enough to be unique.
            pos = norm_narr.find(norm_seg[:50], cursor)
            if pos >= 0:
                scene["start_ms"] = int(pos / total * audio_duration_ms)
                cursor = pos + max(len(seg), 1)
            else:
                scene["start_ms"] = int(cursor / total * audio_duration_ms)
        for i in range(len(scenes) - 1):
            scenes[i]["end_ms"] = scenes[i + 1]["start_ms"]
        scenes[-1]["end_ms"] = int(audio_duration_ms)

    # ── Strategy 3: even distribution ────────────────────────────────────────
    if not word_boundaries and not narration:
        per = audio_duration_ms / n
        for i, scene in enumerate(scenes):
            scene["start_ms"] = int(i * per)
            scene["end_ms"]   = int((i + 1) * per)
        return scenes

    # ── Strategy 2: character-position (no word boundaries) ──────────────────
    if not word_boundaries:
        _by_char_pos()
        return scenes

    # ── Strategy 1: word-boundary monotone matching ───────────────────────────
    wb_words = [b["word"].lower().strip(".,!?;:'\"") for b in word_boundaries]
    wb_times = [b["offset_ms"] for b in word_boundaries]

    search_from = 0
    matched     = 0
    for i, scene in enumerate(scenes):
        seg_words = [
            w.lower().strip(".,!?;:'\"")
            for w in scene.get("text_segment", "").split()
        ]
        # 5 words: enough to be unique even for common short phrases.
        idx = _find_word_sequence(wb_words, seg_words[:5], search_from)
        if idx >= 0:
            scene["start_ms"] = int(wb_times[idx])
            search_from = idx + 1
            matched += 1
        else:
            scene["start_ms"] = int(i * audio_duration_ms / n)

    # If fewer than half the scenes matched, fall back to character-position.
    if matched < len(scenes) // 2 and narration:
        print(f"  Note: only {matched}/{len(scenes)} scenes matched word "
              f"boundaries — using character-position timestamps instead")
        _by_char_pos()
        return scenes

    for i in range(len(scenes) - 1):
        scenes[i]["end_ms"] = scenes[i + 1]["start_ms"]
    last = word_boundaries[-1]
    scenes[-1]["end_ms"] = int(last["offset_ms"] + last["duration_ms"])

    return scenes


def compose_video(
    image_paths: list,
    scenes: list,
    audio_path: str,
    output_path: str = "output/final_video.mp4",
) -> str:
    """
    Compose a timestamp-synced Ken Burns slideshow at 1920×1080.

    image_paths : ordered PNG paths (one per scene)
    scenes      : scene dicts with start_ms / end_ms
    audio_path  : TTS narration audio
    output_path : destination MP4
    """
    video_w, video_h = 1920, 1080
    audio    = AudioFileClip(audio_path)
    total_ms = audio.duration * 1000

    print(f"  {len(image_paths)} scenes  →  {audio.duration:.1f}s  ({video_w}×{video_h} @ {FPS}fps)")

    clips = []
    for i, (path, scene) in enumerate(zip(image_paths, scenes)):
        start_ms = scene.get("start_ms", i * total_ms / len(scenes))
        end_ms   = scene.get("end_ms",   (i + 1) * total_ms / len(scenes))
        duration = max(0.5, (end_ms - start_ms) / 1000.0)

        dur_padded = duration + CROSSFADE_S if i < len(image_paths) - 1 else duration
        clip = _kenburns_clip(path, dur_padded, video_w, video_h)
        if i > 0:
            clip = clip.crossfadein(CROSSFADE_S)
        clips.append(clip)
        print(f"  Clip {i+1}/{len(image_paths)}: {duration:.1f}s", end="\r")

    print()

    final = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE_S)
    final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=4,
        logger="bar",
    )
    audio.close()
    return output_path
