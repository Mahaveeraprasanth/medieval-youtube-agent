"""
Chronicle Forge — Flow Image Importer

Instead of generating images locally with SDXL, this version uses
images that were already generated in Google Flow.

Expected source folder:
    assets/images/

Expected filenames:
    scene_001.jpg
    scene_002.jpg
    scene_003.jpg
    ...

Images are copied to:
    output/images/scene_001.png
    output/images/scene_002.png
    ...

This keeps the existing main.py and video_composer.py compatible.
"""

import os
import shutil
from PIL import Image


SOURCE_DIR = "assets/images"
OUTPUT_DIR = "output/images"


def _find_source_image(index: int) -> str | None:
    """
    Find the Flow image for a scene.

    Supports:
        scene_001.jpg
        scene_001.jpeg
        scene_001.png
        scene_001.webp
    """

    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    filename_base = f"scene_{index:03d}"

    for ext in extensions:
        path = os.path.join(SOURCE_DIR, filename_base + ext)

        if os.path.exists(path):
            return path

    return None


def _convert_to_png(source_path: str, output_path: str) -> None:
    """
    Convert Flow image to PNG.

    PNG output keeps compatibility with the existing Chronicle Forge
    video composer.
    """

    with Image.open(source_path) as img:

        # Handle images with transparency correctly.
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, "white")
            background.paste(img, mask=img.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")

        img.save(output_path, "PNG", optimize=True)


def generate_images(
    scenes: list,
    style_mode: str = "refined",
    verbose: bool = False,
) -> list:
    """
    Import pre-generated Google Flow images.

    The function keeps the same interface expected by main.py.

    One Flow image is required for each scene.
    """

    del style_mode  # Kept for compatibility with main.py.

    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(scenes)

    print()
    print("  Google Flow image importer")
    print(f"  Looking for {total} scene images in: {SOURCE_DIR}")
    print()

    image_paths = []

    for i, scene in enumerate(scenes, start=1):

        source_path = _find_source_image(i)

        if source_path is None:
            raise FileNotFoundError(
                f"\nMissing Flow image for scene {i}.\n\n"
                f"Expected one of:\n"
                f"  {SOURCE_DIR}/scene_{i:03d}.jpg\n"
                f"  {SOURCE_DIR}/scene_{i:03d}.jpeg\n"
                f"  {SOURCE_DIR}/scene_{i:03d}.png\n"
                f"  {SOURCE_DIR}/scene_{i:03d}.webp\n\n"
                f"Please add the missing Google Flow image and run again."
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"scene_{i:03d}.png"
        )

        # Don't reconvert an image that already exists.
        if not os.path.exists(output_path):

            _convert_to_png(
                source_path,
                output_path
            )

            print(
                f"  Imported {i}/{total}: "
                f"{os.path.basename(source_path)}"
            )

        else:

            if verbose:
                print(
                    f"  Existing {i}/{total}: "
                    f"{os.path.basename(output_path)}"
                )

        image_paths.append(output_path)

    print()
    print(f"  {len(image_paths)} Flow images ready.")

    return image_paths


def generate_thumbnail(
    prompt: str,
    headline: str,
    style_mode: str = "refined",
) -> str:
    """
    Create a simple thumbnail from the first available Flow image.

    Google Flow remains responsible for creating the actual thumbnail
    artwork. This function only provides a compatible fallback so that
    main.py does not fail.
    """

    del prompt
    del style_mode

    os.makedirs("output", exist_ok=True)

    thumbnail_path = "output/thumbnail.png"

    # If a thumbnail already exists, leave it alone.
    if os.path.exists(thumbnail_path):
        print(f"  Thumbnail already exists: {thumbnail_path}")
        return thumbnail_path

    # Use scene_001 as the fallback thumbnail source.
    source_path = _find_source_image(1)

    if source_path is None:
        print(
            "  Warning: no scene_001 image available. "
            "Thumbnail was not created."
        )
        return thumbnail_path

    with Image.open(source_path) as img:

        img = img.convert("RGB")

        # YouTube thumbnail ratio: 16:9.
        target_ratio = 16 / 9
        current_ratio = img.width / img.height

        if current_ratio > target_ratio:

            # Image is too wide.
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2

            img = img.crop(
                (left, 0, left + new_width, img.height)
            )

        else:

            # Image is too tall.
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2

            img = img.crop(
                (0, top, img.width, top + new_height)
            )

        img = img.resize(
            (1280, 720),
            Image.Resampling.LANCZOS
        )

        img.save(
            thumbnail_path,
            "PNG",
            optimize=True
        )

    print(f"  Thumbnail created: {thumbnail_path}")

    return thumbnail_path
