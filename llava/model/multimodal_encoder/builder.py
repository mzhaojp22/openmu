import os
from .clip_encoder import CLIPVisionTower
from .audiomae_encoder import AudioMAEencoder


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )  # openai/clip-vit-large-patch14-336
    is_absolute_path_exists = os.path.exists(vision_tower)
    if (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")


def build_audio_tower(audio_tower_cfg, **kwargs):
    # audiomae loading etc.
    return AudioMAEencoder(audio_tower_cfg, **kwargs)
