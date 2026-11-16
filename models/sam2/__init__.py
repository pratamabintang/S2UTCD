# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

try:
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # configs live in models/sam2_configs (sibling of this package) and use
    # ``models.sam2.*`` targets, so compose/instantiate works from the repo root.
    _CONFIG_DIR = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'sam2_configs')
    )
    if not GlobalHydra.instance().is_initialized() and os.path.isdir(_CONFIG_DIR):
        initialize_config_dir(config_dir=_CONFIG_DIR, version_base='1.2')
except Exception:  # hydra/config module is optional when the trunk is built directly
    pass
