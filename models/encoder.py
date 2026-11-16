import torch
import torch.nn as nn

try:
    from .sam2.modeling.backbones.hieradet import Hiera
except Exception:  # pragma: no cover - sam2 trunk is optional
    Hiera = None


class Adapter(nn.Module):
    """Prompt adapter injected after every frozen SAM2 trunk block.

    Zero-initialising the last projection (plus the small ``scale``) makes the
    adapter an exact no-op at step 0, so training starts from the frozen trunk.
    """

    def __init__(self, blk, scale: float = 0.1) -> None:
        super().__init__()
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
        return self.block(prompted)


class SAMEncoder(nn.Module):
    """
    SAM2 (Hiera) image encoder, adapted to the encoder contract used by FewShotSeg:
    it returns a feature dict plus a single global gate/threshold value.

    forward(x) -> (features, t)
        features["layer1".."layer4"]: channels 144/288/576/1152 at strides 4/8/16/32
                                      (shallow -> deep, FewShotSeg order)
        t: (B, 1) global confidence token, analogous to a cls/gate head

    ``channel_list`` keeps SAM2's native deepest-first order (1152/576/288/144),
    like ``models/SAM2UNet.py``.

    The trunk is frozen; only the injected prompt adapters and the threshold head train.
    """

    _HIERA_CFGS = {
        't': dict(embed_dim=96, num_heads=1, stages=[1, 2, 7, 2], global_att_blocks=[5, 7, 9]),
        's': dict(embed_dim=96, num_heads=1, stages=[1, 2, 11, 2], global_att_blocks=[7, 10, 13]),
        'b+': dict(embed_dim=112, num_heads=2),
        'l': dict(embed_dim=144, num_heads=2, stages=[2, 6, 36, 4],
                  global_att_blocks=[23, 33, 43], window_spec=[8, 4, 16, 8]),
    }

    def __init__(self, pretrained_weights=None, model_size='l', freeze=True, use_adapter=True):
        super().__init__()
        if Hiera is None:
            raise ImportError("SAM2 trunk unavailable: could not import "
                              "models.sam2.modeling.backbones.hieradet.Hiera")
        if model_size not in self._HIERA_CFGS:
            raise ValueError(f"Unknown SAM2 size '{model_size}', expected one of {list(self._HIERA_CFGS)}")

        trunk = Hiera(window_pos_embed_bkg_spatial_size=[7, 7], **self._HIERA_CFGS[model_size])
        # SAM2 reports deepest-first; FewShotSeg expects shallow-to-deep levels.
        self.channel_list = list(trunk.channel_list)
        self.out_channels = self.channel_list[::-1]

        if freeze:
            for param in trunk.parameters():
                param.requires_grad = False

        if use_adapter:
            trunk.blocks = nn.Sequential(*[Adapter(block) for block in trunk.blocks])

        self.backbone = trunk
        self.reduce1d = nn.Linear(in_features=self.out_channels[-1], out_features=1, bias=True)

        if pretrained_weights:
            self._init_weights(pretrained_weights)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        features = {"layer1": x1, "layer2": x2, "layer3": x3, "layer4": x4}

        # feature map -> global average pool -> single gate value
        t = x4.flatten(2).mean(dim=-1)
        t = self.reduce1d(t)
        return (features, t)

    def _init_weights(self, checkpoint_path):
        """Load the SAM2 pretrained trunk weights and keep any task-specific heads."""
        weights = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(weights, dict) and isinstance(weights.get('model'), dict):
            weights = weights['model']

        # normalise checkpoint keys, e.g. "model.image_encoder.trunk.blocks.0.attn..." -> "blocks.0.attn..."
        prefixes = ('image_encoder.trunk.', 'backbone.', 'encoder.')
        src = {}
        for key, value in weights.items():
            name = key
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            src[name] = value

        new_dic = self.state_dict()
        loaded = 0
        for key in list(new_dic):
            if key.startswith('reduce1d.'):
                continue  # task head, not part of the SAM2 checkpoint
            for alias in self._alias_keys(key):
                value = src.get(alias)
                if value is not None and value.shape == new_dic[key].shape:
                    new_dic[key] = value
                    loaded += 1
                    break

        self.load_state_dict(new_dic)
        return loaded

    @staticmethod
    def _alias_keys(key):
        """Yield the checkpoint-style aliases of a model state_dict key."""
        if not key.startswith('backbone.'):
            return
        base = key[len('backbone.'):]
        yield base
        if base.startswith('blocks.'):
            _, stage, rest = base.split('.', 2)
            if rest.startswith('block.'):
                rest = rest[len('block.'):]
            yield f'blocks.{stage}.{rest}'
