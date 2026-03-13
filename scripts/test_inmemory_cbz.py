from io import BytesIO
import importlib.util
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm


def _load_stitch_module():
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "SmartStitchLib.py"
    spec = importlib.util.spec_from_file_location("SmartStitchLib", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SmartStitchLib from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stitch = _load_stitch_module()

CBZ_PATH = Path("inputs") / "Chapter 44.cbz"
SPLIT_HEIGHT = 5000
DETECTION_TYPE = "pixel"
DETECTION_SENSTIVITY = 90
IGNORABLE_PIXELS = 5
SCAN_LINE_STEP = 5

SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tga",
)


def load_images_from_cbz(cbz_path: Path) -> list[Image.Image]:
    if not cbz_path.exists():
        raise FileNotFoundError(f"CBZ file was not found: {cbz_path}")

    images: list[Image.Image] = []
    with ZipFile(cbz_path, "r") as archive:
        image_entries = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)
        )
        for entry in image_entries:
            data = archive.read(entry)
            with Image.open(BytesIO(data)) as source_image:
                images.append(source_image.convert("RGB"))

    return images


def main() -> None:
    input_images = load_images_from_cbz(CBZ_PATH)

    processing_bar = tqdm(total=4, desc="Processing images", unit="step")
    progress_state = {"current": 0}

    def on_progress(step: int, total: int, message: str) -> None:
        if processing_bar.total != total:
            processing_bar.total = total

        completed = min(step, total)
        delta = completed - progress_state["current"]
        if delta > 0:
            processing_bar.update(delta)
            progress_state["current"] = completed

        processing_bar.set_postfix_str(message)

    try:
        output_images = stitch.run_images(
            images=input_images,
            split_height=SPLIT_HEIGHT,
            detection_type=DETECTION_TYPE,
            detection_senstivity=DETECTION_SENSTIVITY,
            ignorable_pixels=IGNORABLE_PIXELS,
            scan_line_step=SCAN_LINE_STEP,
            progress_func=on_progress,
        )
    finally:
        processing_bar.close()

    print(f"input_images: {len(input_images)}")
    print(f"output_images: {len(output_images)}")
    if output_images:
        print(f"first_output_size: {output_images[0].size}")
        print(f"last_output_size: {output_images[-1].size}")

    for image in input_images:
        image.close()
    for image in output_images:
        image.close()


if __name__ == "__main__":
    main()
