import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .clip_encoder_prumerge import CLIPVisionTowerPrumerge
from .clip_encoder_tome import CLIPVisionTowerTome
from .clip_encoder_earlymerge import CLIPVisionTowerEarlyMerge

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            projector_type = getattr(vision_tower_cfg, 'mm_projector_type', 'linear')
            if 'prumerge' in projector_type:
                print("Using Prumerge !!!!!")
                return CLIPVisionTowerPrumerge(vision_tower, args=vision_tower_cfg, **kwargs)
            elif 'tome' in projector_type:
                print("Using Tome !!!!!")
                return CLIPVisionTowerTome(vision_tower, args=vision_tower_cfg, **kwargs)
            elif 'earlymerge' in projector_type:
                print("Using EarlyMerge !!!!!")
                # import ipdb; ipdb.set_trace()
                return CLIPVisionTowerEarlyMerge(vision_tower, args=vision_tower_cfg, **kwargs)
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
