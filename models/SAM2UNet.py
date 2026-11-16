import logging
import os
import sys

import torch
import torch.nn as nn

if __package__ in (None, ''):
    # run as a plain script: put the repo root on sys.path so `models.*` resolves
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.sam2.build_sam import build_sam2
else:
    from .sam2.build_sam import build_sam2


class Adapter(nn.Module):
    def __init__(self, blk, scale: float = 0.1) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        self.scale = scale
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )
        nn.init.zeros_(self.prompt_learn[2].weight)
        nn.init.zeros_(self.prompt_learn[2].bias)

    def forward(self, x):
        prompt = self.prompt_learn(x)
        prompted = x + self.scale * prompt
        net = self.block(prompted)
        return net


class SAM2(nn.Module):
    """SAM2 image-encoder only: keeps the frozen Hiera trunk + prompt adapters."""

    def __init__(self, checkpoint_path: str = None, model_cfg: str = "sam2_hiera_l.yaml") -> None:
        super(SAM2, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path, device=device)
        else:
            logging.warning('SAM2 checkpoint not provided, image encoder uses random weights')
            model = build_sam2(model_cfg, device=device)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk
        self.channel_list = list(self.encoder.channel_list)

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        return {
            '0': x1,
            '1': x2,
            '2': x3,
            '3': x4
        }


if __name__ == "__main__":
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SAM2(model_cfg="sam2_hiera_t.yaml").to(device)
        x = torch.randn(1, 3, 352, 352, device=device)
        out = model(x)
        print(out['0'].shape, out['1'].shape, out['2'].shape, out['3'].shape)
