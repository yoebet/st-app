from dataclasses import dataclass
from typing import Any, Optional, Dict, List


@dataclass(kw_only=True)
class Task:
    task_id: str  # xd43w
    image_url: str
    audio_url: str
    subdir: Optional[str] = None  # 2024-01-08
    style_name: Optional[str] = None
    pose_name: Optional[str] = None
    cfg_scale: Optional[float] = None  # 2.0
    max_gen_len: Optional[int] = None  # seconds


@dataclass(kw_only=True)
class LaunchOptions:
    device_index: Optional[int] = None
    proxy: Optional[str] = None  # pc/http/clear
    hf_hub_offline: Optional[bool] = None


@dataclass(kw_only=True)
class TaskParams(Task, LaunchOptions):
    task_dir: Optional[str] = None
    image_path: Optional[str] = None  # male_face.png
    audio_path: Optional[str] = None  # acknowledgement_english.m4a
    style_clip_path: Optional[str] = None  # data/style_clip/3DMM/M030_front_neutral_level1_001.mat
    pose_path: Optional[str] = None  # data/pose/RichardShelby_front_neutral_level1_001.mat
    device: Optional[str] = None
    img_crop: Optional[bool] = None
    cropped_image_path: Optional[str] = None
    output_video_path: Optional[str] = None
