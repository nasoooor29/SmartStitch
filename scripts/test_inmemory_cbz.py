from io import BytesIO
from pathlib import Path
import sys
from zipfile import ZipFile

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SmartStitchLib import split_image

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
    output_images = []
    for image in tqdm(input_images, desc="Processing images", unit="img"):
        split_images = split_image(
            image=image,
            split_height=SPLIT_HEIGHT,
            detection_type=DETECTION_TYPE,
            detection_senstivity=DETECTION_SENSTIVITY,
            ignorable_pixels=IGNORABLE_PIXELS,
            scan_line_step=SCAN_LINE_STEP,
        )
        output_images.extend(split_images)

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
