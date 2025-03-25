import torch
import torch.nn as nn
import json
from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


class ImageEncoderWithMLP(nn.Module):
    def __init__(self, config_path, checkpoint_path, frozen_encoder=True, embed_dim=512):
        super().__init__()
        
        self.frozen_encoder = frozen_encoder


        with open(config_path, "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]


        model_name = "biomedclip_local"
        if model_name not in _MODEL_CONFIGS:
            _MODEL_CONFIGS[model_name] = model_cfg

        self.model, _,self.preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint_path,
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        self.encoder = self.model.visual

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )


    def forward(self, image):
        if self.frozen_encoder:
            with torch.no_grad():
                features = self.encoder(image)
        else:
            features = self.encoder(image)
        out = self.classifier(features)
        return out
    
    def get_preprocess(self):
        return self.preprocess