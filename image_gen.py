"""SDXL image generation for chronicle-forge.

Two-channel SDXL prompt strategy (fixed):
  CLIP ViT-L (prompt)   → concise subject+action, max 20 words.
                          Extracted from the cinematic prompt by cutting at
                          the first structured section marker (, environment:
                          , lighting:, etc.). Stays well within the 77-token
                          budget. Controls WHAT is drawn.

  OpenCLIP ViT-bigG (prompt_2) → concise subject + style keywords, trimmed
                          to 74 tokens using tokenizer_2. Controls HOW it looks.

Root cause of earlier image-relevance failure:
  The 40-60 word cinematic prompts exceeded CLIP's 77-token limit. _trim_clip
  truncated mid-sentence, so CLIP received "industrial yard with broken" instead
  of the subject. OpenCLIP received only ILLUS_STYLE (no scene content at all).
  The fix: CLIP gets the 10-20 word extracted subject; OpenCLIP gets that same
  subject + style keywords (trimmed to 74 tokens).

Two style modes (--style flag in main.py):
  stickman  — simple stick-figure / flat shapes, Zenn YouTube aesthetic,
              white background, bold primary colors, minimal detail.
  refined   — high-quality 2D digital illustration, cel shading, cinematic
              lighting, detailed characters and environments.
"""
import gc
import os
import random
import time

import numpy as np
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"Z:\Programs\Invoke\models\0b629f4f-bf32-4a49-88df-52b8c8c87ec6",
)
STEPS = int(os.getenv("INVOKE_STEPS", "20"))
CFG   = float(os.getenv("INVOKE_CFG",   "7.0"))

IMG_W, IMG_H = 1216, 832   # 16:9 → upscaled to 1920×1080 in video
THUMB_W, THUMB_H = 1280, 720   # standard YouTube thumbnail resolution

# ── Style configs ─────────────────────────────────────────────────────────────

STYLE_CONFIGS = {
    "stickman": {
        # Zenn-channel aesthetic: simple stick figures, white BG, bold flat colors.
        # Juggernaut XL is a photorealistic base that strongly resists illustration styles.
        # To overcome this:
        #   1. clip_prefix forces CLIP ViT-L to see the style intent (not just the subject)
        #   2. guidance_scale = 11 to weight the prompt over the model's base tendency
        #   3. Aggressive negative prompt blocking every photorealistic cue
        "style": (
            "simple 2D stick figure illustration, white background, "
            "bold solid flat colors, minimal character design, "
            "educational whiteboard animation style, thick black outlines, "
            "simple geometric shapes, primary color palette, "
            "clean uncluttered flat layout, no shading, no gradients, "
            "legible text, accurate lettering"
        ),
        "negative": (
            "photorealistic, photograph, real person, realistic face, "
            "realistic skin, detailed anatomy, 3D render, CGI, "
            "complex shading, depth of field, bokeh, "
            "dark background, busy composition, watercolor, oil paint, "
            "detailed texture, realistic lighting, shadows, ugly, low quality, "
            "misspelled text, incorrect numbers, garbled letters, wrong digits, "
            "partial text, illegible writing"
        ),
        "clip_prefix": "simple stick figure illustration of ",
        "guidance_scale": 11.0,
    },
    "refined": {
        # High-quality 2D animation: cel shading, detailed characters, dramatic lighting.
        # Works WITH Juggernaut XL's strength rather than against it.
        "style": (
            "high quality 2D digital illustration, detailed character design, "
            "smooth cel shading, vibrant cinematic colors, "
            "professional concept art, dramatic lighting, "
            "expressive characters, detailed environments, "
            "dynamic composition, studio animation quality, "
            "legible text, accurate lettering"
        ),
        "negative": (
            "photorealistic photograph, realistic skin pores, "
            "bokeh depth of field, photo rendering, "
            "ugly, deformed, blurry, grainy, low quality, "
            "text watermark, unfinished sketch, noisy, "
            "misspelled text, incorrect numbers, garbled letters, wrong digits, "
            "partial text, illegible writing"
        ),
        "clip_prefix": "",
        "guidance_scale": None,   # use INVOKE_CFG from .env
    },
}

DEFAULT_STYLE = "refined"

_pipeline   = None
_dml_device = None


# ── DML patches ───────────────────────────────────────────────────────────────

def _patch_dml_unet(unet, dml_device):
    """
    Make the SDXL UNet a transparent CPU->DML->CPU bridge.

    Root problem: diffusers creates latents and embeddings on CPU (text_encoders
    live on CPU so _execution_device resolves to CPU). The UNet is on DML. Any
    DML module that asserts its input is DML-resident crashes with
    "unbox expects Dml at::Tensor as inputs".

    Patch 0 (bridge): wrap unet.forward so ALL tensor inputs are moved to DML on
    entry and ALL tensor outputs are moved back to CPU on exit. The pipeline,
    scheduler, and latent book-keeping then run entirely on CPU with no changes,
    while every internal UNet op runs on DML as intended.

    Patches 1-3 handle CPU tensors produced INSIDE the UNet forward pass itself
    (int64 timestep, sinusoidal float, SDXL add_embeds).
    """
    import torch as _torch

    # Patch 0: CPU->DML->CPU bridge on unet.forward
    orig_unet_fwd = unet.forward

    def _bridged_unet_fwd(sample, timestep, encoder_hidden_states, *args, **kwargs):
        sample = sample.to(dml_device)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dml_device)
        if kwargs.get("added_cond_kwargs"):
            kwargs["added_cond_kwargs"] = {
                k: v.to(dml_device) if isinstance(v, _torch.Tensor) else v
                for k, v in kwargs["added_cond_kwargs"].items()
            }
        result = orig_unet_fwd(sample, timestep, encoder_hidden_states, *args, **kwargs)
        if isinstance(result, tuple):
            return tuple(
                x.to("cpu") if isinstance(x, _torch.Tensor) else x for x in result
            )
        if isinstance(result, _torch.Tensor):
            return result.to("cpu")
        return result

    unet.forward = _bridged_unet_fwd

    # Patch 1: time_proj — int64 timestep cannot be computed on DML; run on CPU
    orig_time_proj = unet.time_proj.forward

    def _cpu_time_proj(timesteps):
        return orig_time_proj(timesteps.cpu())

    unet.time_proj.forward = _cpu_time_proj

    # Patch 2: time_embedding — receives CPU float from time_proj; move to DML
    orig_time_emb = unet.time_embedding.forward

    def _dml_time_emb(*args, **kwargs):
        new_args = (args[0].to(dml_device),) + args[1:]
        return orig_time_emb(*new_args, **kwargs)

    unet.time_embedding.forward = _dml_time_emb

    # Patch 3: add_embedding (SDXL) — add_embeds concat happens before this; move to DML
    if hasattr(unet, "add_embedding"):
        orig_add_emb = unet.add_embedding.forward

        def _dml_add_emb(*args, **kwargs):
            new_args = (args[0].to(dml_device),) + args[1:]
            return orig_add_emb(*new_args, **kwargs)

        unet.add_embedding.forward = _dml_add_emb


def _load_pipeline():
    global _pipeline, _dml_device
    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        import torch_directml
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
    except ImportError as exc:
        raise RuntimeError(
            f"Missing package: {exc}\n"
            "Run: pip install torch-directml diffusers transformers accelerate safetensors"
        ) from exc

    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\nCheck MODEL_PATH in .env"
        )

    import torch
    import torch_directml
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    print("  Loading SDXL (UNet→DirectML | VAE→CPU)...", end=" ", flush=True)
    t0 = time.time()

    dml  = torch_directml.device()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )
    pipe.unet.to(dml)
    pipe.vae.to(torch.float32)
    pipe.enable_attention_slicing(1)

    _patch_dml_unet(pipe.unet, dml)

    # Force prepare_latents to always create CPU latents.
    # See arch.md for the full scheduler device-mismatch explanation.
    _orig_prepare = pipe.prepare_latents

    def _cpu_prepare_latents(
        batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size, num_channels_latents,
            int(height) // pipe.vae_scale_factor,
            int(width) // pipe.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(shape, generator=generator, dtype=dtype)
        return latents * pipe.scheduler.init_noise_sigma

    pipe.prepare_latents = _cpu_prepare_latents

    _dml_device = dml
    print(f"done ({time.time() - t0:.0f}s)")
    _pipeline = pipe
    return pipe


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _extract_clip_subject(prompt: str) -> str:
    """
    Extract the subject+action portion from a cinematic prompt for CLIP ViT-L.

    Pass 2 cinematic prompts follow the pattern:
      [subject+action], [environment], [lighting], [camera angle], [mood]

    CLIP ViT-L has a hard 77-token limit. With 40-60 word full prompts (~60-90
    tokens), the model truncates mid-sentence and loses semantic meaning. By
    extracting only the subject+action part (~15-20 words / ~20-25 tokens),
    CLIP gets a clean, unambiguous description of WHAT to draw.
    """
    for marker in [", environment", " environment:", ", lighting", " lighting:",
                   ", camera", " camera:", ", emotional tone", " emotional"]:
        idx = prompt.lower().find(marker.lower())
        if idx > 20:
            return prompt[:idx].strip().rstrip(",").strip()
    # No structured marker found — take first 60 characters, break at word boundary
    if len(prompt) > 60:
        return prompt[:60].rsplit(" ", 1)[0]
    return prompt


def _trim_clip(pipe, text: str, limit: int = 74) -> str:
    """Trim to 74 CLIP tokens (pipe.tokenizer) — leaves 3 tokens for BOS/EOS/PAD."""
    ids = pipe.tokenizer.encode(text)
    if len(ids) <= limit:
        return text
    return pipe.tokenizer.decode(ids[:limit], skip_special_tokens=True)


def _trim_openclip(pipe, text: str, limit: int = 74) -> str:
    """Trim to 74 OpenCLIP tokens (pipe.tokenizer_2) — same budget, different vocab."""
    ids = pipe.tokenizer_2.encode(text)
    if len(ids) <= limit:
        return text
    return pipe.tokenizer_2.decode(ids[:limit], skip_special_tokens=True)


def _decode_on_cpu(pipe, latents) -> PILImage.Image:
    """VAE decode on CPU to avoid VRAM OOM from the large scratch buffer."""
    import torch
    latents_cpu = latents.to("cpu").float()
    with torch.no_grad():
        decoded = pipe.vae.decode(
            latents_cpu / pipe.vae.config.scaling_factor
        ).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    arr     = decoded.squeeze(0).permute(1, 2, 0).numpy()
    arr     = (arr * 255).round().astype(np.uint8)
    return PILImage.fromarray(arr)


# ── Thumbnail helpers ─────────────────────────────────────────────────────────

def _draw_thumbnail_text(image: PILImage.Image, headline: str) -> PILImage.Image:
    """Overlay bold Impact-style headline on the thumbnail with a dark bar."""
    from PIL import ImageDraw, ImageFont
    import textwrap

    w, h      = image.size
    font_size = 80
    font      = None

    for fp in [
        r"C:\Windows\Fonts\impact.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                pass
    if font is None:
        font      = ImageFont.load_default()
        font_size = 20

    lines   = textwrap.wrap(headline.upper(), width=20)[:2]
    measure = ImageDraw.Draw(PILImage.new("RGB", (1, 1)))

    line_ws, line_hs = [], []
    for line in lines:
        bb = measure.textbbox((0, 0), line, font=font)
        line_ws.append(bb[2] - bb[0])
        line_hs.append(bb[3] - bb[1])

    spacing  = 10
    total_h  = sum(line_hs) + spacing * max(0, len(lines) - 1)
    bar_pad  = 24
    bar_h    = total_h + bar_pad * 2
    bar_top  = h - bar_h - 16

    overlay   = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
    bar_draw  = ImageDraw.Draw(overlay)
    bar_draw.rectangle([(0, bar_top), (w, h)], fill=(0, 0, 0, 175))
    image     = PILImage.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw      = ImageDraw.Draw(image)

    y       = bar_top + bar_pad
    outline = 3
    for line, lw, lh in zip(lines, line_ws, line_hs):
        x = (w - lw) // 2
        for ox, oy in [(-outline, 0), (outline, 0), (0, -outline), (0, outline),
                       (-outline, -outline), (outline, -outline),
                       (-outline, outline), (outline, outline)]:
            draw.text((x + ox, y + oy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += lh + spacing

    return image


def generate_thumbnail(
    image_prompt: str,
    headline: str,
    style_mode: str = DEFAULT_STYLE,
) -> str:
    """
    Generate a YouTube thumbnail at 1280×720 with headline text overlay.

    Uses the same SDXL pipeline as scene images (loads it if not already warm).
    Adds text-render blocking to the negative prompt so SDXL doesn't try to
    draw its own letters. Saves to output/thumbnail.png and returns the path.
    """
    config      = STYLE_CONFIGS.get(style_mode, STYLE_CONFIGS[DEFAULT_STYLE])
    style_str   = config["style"]
    clip_prefix = config.get("clip_prefix", "")
    cfg         = config.get("guidance_scale") or CFG

    # Block SDXL from rendering text — we overlay it ourselves
    negative_str = (
        config["negative"]
        + ", text, letters, words, caption, watermark, subtitle, title card, speech bubble"
    )

    pipe = _load_pipeline()
    import torch

    print(f"  Thumbnail: {image_prompt[:60]}...")
    clip_prompt = _trim_clip(pipe, f"{clip_prefix}{image_prompt}")
    openclip    = _trim_openclip(pipe, f"{image_prompt}, {style_str}")

    seed      = random.randint(0, 2**31 - 1)
    generator = torch.Generator("cpu").manual_seed(seed)
    cpu       = torch.device("cpu")

    (prompt_emb, neg_emb,
     pooled_emb, neg_pooled_emb) = pipe.encode_prompt(
        prompt=clip_prompt,
        prompt_2=openclip,
        device=cpu,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_str,
        negative_prompt_2=negative_str,
    )

    for attempt in range(2):
        try:
            out = pipe(
                prompt_embeds=prompt_emb,
                negative_prompt_embeds=neg_emb,
                pooled_prompt_embeds=pooled_emb,
                negative_pooled_prompt_embeds=neg_pooled_emb,
                width=THUMB_W,
                height=THUMB_H,
                num_inference_steps=STEPS,
                guidance_scale=cfg,
                generator=generator,
                output_type="latent",
            )
            break
        except RuntimeError as exc:
            if attempt == 0:
                print(f"\n  DML error (attempt 1) — retrying... ({exc})")
                gc.collect()
            else:
                raise

    image = _decode_on_cpu(pipe, out.images)
    del out, prompt_emb, neg_emb, pooled_emb, neg_pooled_emb
    gc.collect()

    image = _draw_thumbnail_text(image, headline)

    dest = "output/thumbnail.png"
    os.makedirs("output", exist_ok=True)
    image.save(dest, optimize=True)
    print(f"  Thumbnail saved → {dest}")
    return dest


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_images(scenes: list, style_mode: str = DEFAULT_STYLE,
                    verbose: bool = False) -> list:
    """
    Generate one 16:9 image per scene.

    Prompt strategy (see module docstring for full explanation):
      CLIP ViT-L    ← concise subject+action extracted from cinematic prompt
      OpenCLIP bigG ← same subject + style keywords (trimmed to 74 tokens)

    style_mode: "stickman" (Zenn-style flat) or "refined" (quality 2D animation)

    Skips scenes whose output PNG already exists (size > 1 KB) for resume support.
    """
    config       = STYLE_CONFIGS.get(style_mode, STYLE_CONFIGS[DEFAULT_STYLE])
    style_str    = config["style"]
    negative_str = config["negative"]
    clip_prefix  = config.get("clip_prefix", "")
    cfg          = config.get("guidance_scale") or CFG

    os.makedirs("output/images", exist_ok=True)
    pipe = _load_pipeline()   # imports torch inside; raises RuntimeError with install hint if missing
    import torch              # safe here — _load_pipeline already populated sys.modules

    print(f"  Style mode: {style_mode}  |  CFG: {cfg}")

    n     = len(scenes)
    paths = []

    for i, scene in enumerate(scenes):
        dest = f"output/images/scene_{i+1:03d}.png"

        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"  Scene {i+1}/{n}: skipping (exists)")
            paths.append(dest)
            continue

        full_prompt = scene.get("prompt", scene.get("visual_subject", ""))

        # CLIP ViT-L: concise subject + optional style prefix (stickman mode injects
        # the style intent here so CLIP also sees it, not just the subject).
        clip_subject = _extract_clip_subject(full_prompt)
        clip_prompt  = _trim_clip(pipe, f"{clip_prefix}{clip_subject}")

        # OpenCLIP ViT-bigG: subject + style keywords — trimmed by its own tokenizer
        openclip = _trim_openclip(pipe, f"{clip_subject}, {style_str}")

        seed = random.randint(0, 2**31 - 1)

        if verbose:
            print(f"\n  ┌─ Scene {i+1}/{n} ──────────────────────────────────")
            print(f"  │ Narration:  {scene.get('text_segment', '')[:70]}")
            print(f"  │ CLIP:       {clip_prompt}")
            print(f"  │ OpenCLIP:   {openclip[:80]}")
            print(f"  └────────────────────────────────────────────────────")

        print(f"  Scene {i+1}/{n} → encode...", end=" ", flush=True)
        t0 = time.time()

        generator = torch.Generator("cpu").manual_seed(seed)
        cpu       = torch.device("cpu")

        (prompt_emb, neg_emb,
         pooled_emb, neg_pooled_emb) = pipe.encode_prompt(
            prompt=clip_prompt,
            prompt_2=openclip,
            device=cpu,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_str,
            negative_prompt_2=negative_str,
        )

        t_enc = time.time() - t0
        print(f"done ({t_enc:.0f}s) → diffuse...", end=" ", flush=True)
        t0 = time.time()

        pipe_kwargs = dict(
            prompt_embeds=prompt_emb,
            negative_prompt_embeds=neg_emb,
            pooled_prompt_embeds=pooled_emb,
            negative_pooled_prompt_embeds=neg_pooled_emb,
            width=IMG_W,
            height=IMG_H,
            num_inference_steps=STEPS,
            guidance_scale=cfg,   # per-style CFG (stickman=11, refined=INVOKE_CFG)
            generator=generator,
            output_type="latent",
        )
        for attempt in range(2):
            try:
                out = pipe(**pipe_kwargs)
                break
            except RuntimeError as exc:
                if attempt == 0:
                    print(f"\n  DML error (attempt 1) — flushing and retrying... ({exc})")
                    gc.collect()
                else:
                    raise

        t_diff = time.time() - t0
        print(f"done ({t_diff:.0f}s) → decode...", end=" ", flush=True)
        t1 = time.time()

        image = _decode_on_cpu(pipe, out.images)

        del out, prompt_emb, neg_emb, pooled_emb, neg_pooled_emb, pipe_kwargs
        gc.collect()

        try:
            image.save(dest)
        except KeyboardInterrupt:
            if os.path.exists(dest):
                os.remove(dest)
            raise

        paths.append(dest)
        print(f"done ({time.time() - t1:.0f}s)  → {dest}")

    return paths
