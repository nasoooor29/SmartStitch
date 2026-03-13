import gc
from dataclasses import dataclass, field
from time import time
from typing import Callable

from PIL import Image as pil


@dataclass(slots=True)
class StitchSettings:
    input_folder: str
    split_height: int
    output_type: str = ".png"
    custom_width: int = -1
    detection_type: str = "pixel"
    detection_senstivity: int = 90
    lossy_quality: int = 100
    ignorable_pixels: int = 5
    scan_line_step: int = 5
    output_folder: str | None = None
    postprocess_folder: str | None = None


@dataclass(slots=True)
class StitchedDirectory:
    input_path: str
    input_files: list[str] = field(default_factory=list)
    output_path: str = ""
    output_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StitchResult:
    directories: list[StitchedDirectory]
    elapsed_seconds: float

    @property
    def total_directories(self) -> int:
        return len(self.directories)

    @property
    def total_output_files(self) -> int:
        return sum(len(directory.output_files) for directory in self.directories)


@dataclass(slots=True)
class InMemoryStitchSettings:
    split_height: int
    custom_width: int = -1
    detection_type: str = "pixel"
    detection_senstivity: int = 90
    ignorable_pixels: int = 5
    scan_line_step: int = 5


class SmartStitch:
    def run(
        self,
        settings: StitchSettings,
        status_func: Callable[[str], None] = print,
    ) -> StitchResult:
        from core.detectors import select_detector
        from core.services import DirectoryExplorer, ImageHandler, ImageManipulator
        from core.utils.constants import WIDTH_ENFORCEMENT

        self._validate_settings(settings)
        explorer = DirectoryExplorer()
        img_handler = ImageHandler()
        img_manipulator = ImageManipulator()

        detector = select_detector(detection_type=settings.detection_type)
        width_enforce_mode = (
            WIDTH_ENFORCEMENT.MANUAL
            if settings.custom_width > 0
            else WIDTH_ENFORCEMENT.NONE
        )
        explorer_kwargs = {}
        if settings.output_folder:
            explorer_kwargs["output"] = settings.output_folder
        if settings.postprocess_folder:
            explorer_kwargs["postprocess"] = settings.postprocess_folder

        start_time = time()
        status_func("Exploring input directory for working directories")
        input_dirs = explorer.run(input=settings.input_folder, **explorer_kwargs)
        input_dirs_count = len(input_dirs)
        status_func(f"[{input_dirs_count}] Working directories were found")

        dir_iteration = 1
        output_directories = []
        for workdirectory in input_dirs:
            status_func(
                f"[{dir_iteration}/{input_dirs_count}] Preparing & loading images into memory"
            )
            imgs = img_handler.load(workdirectory)
            imgs = img_manipulator.resize(
                imgs, width_enforce_mode, settings.custom_width
            )

            status_func(
                f"[{dir_iteration}/{input_dirs_count}] Combining images into a single image"
            )
            combined_img = img_manipulator.combine(imgs)

            status_func(
                f"[{dir_iteration}/{input_dirs_count}] Detecting valid slicing points"
            )
            slice_points = detector.run(
                combined_img,
                settings.split_height,
                sensitivity=settings.detection_senstivity,
                ignorable_pixels=settings.ignorable_pixels,
                scan_step=settings.scan_line_step,
            )

            status_func(
                f"[{dir_iteration}/{input_dirs_count}] Generating sliced output images"
            )
            sliced_imgs = img_manipulator.slice(combined_img, slice_points)

            status_func(
                f"[{dir_iteration}/{input_dirs_count}] Saving output images to storage"
            )
            img_iteration = 1
            for img in sliced_imgs:
                img_handler.save(
                    workdirectory,
                    img,
                    img_iteration,
                    img_format=settings.output_type,
                    quality=settings.lossy_quality,
                )
                img_iteration += 1

            output_directories.append(
                StitchedDirectory(
                    input_path=workdirectory.input_path,
                    input_files=list(workdirectory.input_files),
                    output_path=workdirectory.output_path,
                    output_files=list(workdirectory.output_files),
                )
            )

            dir_iteration += 1
            gc.collect()

        elapsed_seconds = time() - start_time
        status_func(f"Process completed in {elapsed_seconds:.3f} seconds")
        return StitchResult(output_directories, elapsed_seconds)

    def run_images(
        self,
        images: list[pil.Image],
        settings: InMemoryStitchSettings,
    ) -> list[pil.Image]:
        self._validate_inmemory_settings(settings)
        if not images:
            raise ValueError("images must contain at least one PIL image")

        output_images = []
        for image in images:
            split_images = self.split_image(image=image, settings=settings)
            output_images.extend(split_images)
        return output_images

    def split_image(
        self,
        image: pil.Image,
        settings: InMemoryStitchSettings,
    ) -> list[pil.Image]:
        self._validate_inmemory_settings(settings)
        if image is None:
            raise ValueError("image is required")

        working_images = [image.copy()]
        working_images = self._resize_images(working_images, settings.custom_width)
        if not working_images:
            return []

        working_image = working_images[0]
        if working_image.mode != "RGB":
            working_image = working_image.convert("RGB")

        slice_points = self._detect_slice_points(
            combined_img=working_image,
            split_height=settings.split_height,
            detection_type=settings.detection_type,
            sensitivity=settings.detection_senstivity,
            ignorable_pixels=settings.ignorable_pixels,
            scan_step=settings.scan_line_step,
        )
        return self._slice_image(working_image, slice_points)

    @staticmethod
    def _resize_images(images: list[pil.Image], custom_width: int) -> list[pil.Image]:
        if custom_width <= 0:
            return images
        resample = getattr(pil, "Resampling", pil).LANCZOS
        resized_images = []
        for image in images:
            if image.size[0] == custom_width:
                resized_images.append(image)
                continue
            img_ratio = float(image.size[1] / image.size[0])
            new_img_height = int(img_ratio * custom_width)
            if new_img_height <= 0:
                continue
            resized_images.append(
                image.resize((custom_width, new_img_height), resample)
            )
        return resized_images

    @staticmethod
    def _combine_images(images: list[pil.Image]) -> pil.Image:
        widths, heights = zip(*(img.size for img in images))
        combined_width = max(widths)
        combined_height = sum(heights)
        combined_img = pil.new("RGB", (combined_width, combined_height))
        combine_offset = 0
        for image in images:
            img_to_paste = image.convert("RGB") if image.mode != "RGB" else image
            combined_img.paste(img_to_paste, (0, combine_offset))
            combine_offset += image.size[1]
        return combined_img

    @staticmethod
    def _slice_image(
        combined_img: pil.Image, slice_locations: list[int]
    ) -> list[pil.Image]:
        max_width = combined_img.size[0]
        sliced_images = []
        for index in range(1, len(slice_locations)):
            upper_limit = slice_locations[index - 1]
            lower_limit = slice_locations[index]
            slice_boundaries = (0, upper_limit, max_width, lower_limit)
            sliced_images.append(combined_img.crop(slice_boundaries))
        combined_img.close()
        return sliced_images

    @staticmethod
    def _detect_slice_points(
        combined_img: pil.Image,
        split_height: int,
        detection_type: str,
        sensitivity: int,
        ignorable_pixels: int,
        scan_step: int,
    ) -> list[int]:
        last_row = combined_img.size[1]
        if detection_type == "none":
            slice_locations = [0]
            row = split_height
            while row < last_row:
                slice_locations.append(row)
                row += split_height
            if slice_locations[-1] != last_row - 1:
                slice_locations.append(last_row - 1)
            return slice_locations

        if detection_type != "pixel":
            raise ValueError("detection_type must be either 'pixel' or 'none'")

        gray_img = combined_img.convert("L")
        threshold = int(255 * (1 - (sensitivity / 100)))
        slice_locations = [0]
        row = split_height
        move_up = True

        while row < last_row:
            row_pixels = gray_img.crop((0, row, gray_img.size[0], row + 1)).getdata()
            can_slice = True
            for index in range(
                ignorable_pixels + 1, len(row_pixels) - ignorable_pixels
            ):
                prev_pixel = int(row_pixels[index - 1])
                next_pixel = int(row_pixels[index])
                value_diff = next_pixel - prev_pixel
                if value_diff > threshold or value_diff < -threshold:
                    can_slice = False
                    break

            if can_slice:
                slice_locations.append(row)
                row += split_height
                move_up = True
                continue

            if row - slice_locations[-1] <= 0.4 * split_height:
                row = slice_locations[-1] + split_height
                move_up = False

            if move_up:
                row -= scan_step
                continue

            row += scan_step

        if slice_locations[-1] != last_row - 1:
            slice_locations.append(last_row - 1)
        return slice_locations

    @staticmethod
    def _validate_settings(settings: StitchSettings):
        if not settings.input_folder:
            raise ValueError("input_folder is required")
        if settings.split_height <= 0:
            raise ValueError("split_height must be > 0")
        if settings.custom_width != -1 and settings.custom_width <= 0:
            raise ValueError("custom_width must be -1 or > 0")
        if not (0 <= settings.detection_senstivity <= 100):
            raise ValueError("detection_senstivity must be between 0 and 100")
        if not (1 <= settings.scan_line_step <= 100):
            raise ValueError("scan_line_step must be between 1 and 100")
        if not (0 <= settings.lossy_quality <= 100):
            raise ValueError("lossy_quality must be between 0 and 100")
        if settings.ignorable_pixels < 0:
            raise ValueError("ignorable_pixels must be >= 0")

    @staticmethod
    def _validate_inmemory_settings(settings: InMemoryStitchSettings):
        if settings.split_height <= 0:
            raise ValueError("split_height must be > 0")
        if settings.custom_width != -1 and settings.custom_width <= 0:
            raise ValueError("custom_width must be -1 or > 0")
        if not (0 <= settings.detection_senstivity <= 100):
            raise ValueError("detection_senstivity must be between 0 and 100")
        if not (1 <= settings.scan_line_step <= 100):
            raise ValueError("scan_line_step must be between 1 and 100")
        if settings.ignorable_pixels < 0:
            raise ValueError("ignorable_pixels must be >= 0")


def run(
    input_folder: str,
    split_height: int,
    output_type: str = ".png",
    custom_width: int = -1,
    detection_type: str = "pixel",
    detection_senstivity: int = 90,
    lossy_quality: int = 100,
    ignorable_pixels: int = 5,
    scan_line_step: int = 5,
    output_folder: str | None = None,
    postprocess_folder: str | None = None,
    status_func: Callable[[str], None] = print,
) -> StitchResult:
    settings = StitchSettings(
        input_folder=input_folder,
        split_height=split_height,
        output_type=output_type,
        custom_width=custom_width,
        detection_type=detection_type,
        detection_senstivity=detection_senstivity,
        lossy_quality=lossy_quality,
        ignorable_pixels=ignorable_pixels,
        scan_line_step=scan_line_step,
        output_folder=output_folder,
        postprocess_folder=postprocess_folder,
    )
    return SmartStitch().run(settings=settings, status_func=status_func)


def run_images(
    images: list[pil.Image],
    split_height: int,
    custom_width: int = -1,
    detection_type: str = "pixel",
    detection_senstivity: int = 90,
    ignorable_pixels: int = 5,
    scan_line_step: int = 5,
) -> list[pil.Image]:
    settings = InMemoryStitchSettings(
        split_height=split_height,
        custom_width=custom_width,
        detection_type=detection_type,
        detection_senstivity=detection_senstivity,
        ignorable_pixels=ignorable_pixels,
        scan_line_step=scan_line_step,
    )
    return SmartStitch().run_images(
        images=images,
        settings=settings,
    )


def split_image(
    image: pil.Image,
    split_height: int,
    custom_width: int = -1,
    detection_type: str = "pixel",
    detection_senstivity: int = 90,
    ignorable_pixels: int = 5,
    scan_line_step: int = 5,
) -> list[pil.Image]:
    settings = InMemoryStitchSettings(
        split_height=split_height,
        custom_width=custom_width,
        detection_type=detection_type,
        detection_senstivity=detection_senstivity,
        ignorable_pixels=ignorable_pixels,
        scan_line_step=scan_line_step,
    )
    return SmartStitch().split_image(image=image, settings=settings)


__all__ = [
    "StitchSettings",
    "StitchedDirectory",
    "StitchResult",
    "InMemoryStitchSettings",
    "SmartStitch",
    "run",
    "run_images",
    "split_image",
]
