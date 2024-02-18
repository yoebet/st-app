import os
from core.utils import crop_src_image


def detect_and_crop(image_path, crop_image_path=None, increase_ratio=0.4):
    image_base, image_ext = os.path.splitext(image_path)
    if crop_image_path is None:
        crop_image_path = f"{image_base}-cropped{image_ext}"
    crop_src_image(image_path, crop_image_path, increase_ratio)
    return crop_image_path


if __name__ == "__main__":
    path = detect_and_crop("../tmp/20231101-173105.jpeg")
    print(path)
