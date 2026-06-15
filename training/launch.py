# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# export PYTHONPATH=$PWD:$PYTHONPATH
# TORCH_NCCL_ASYNC_ERROR_HANDLING=1 torchrun --standalone --nproc_per_node=8 training/launch.py

import os
import sys

# Ensure repo root (panovggt) and training/ are importable without manual PYTHONPATH.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_PROJECT_ROOT, _TRAIN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


