#!/usr/bin/env python
r"""Evaluate FewShotSeg on PASCAL-5i (episodic mIoU).

Example:
    python test_pascal.py --datapath ./data/VOCdevkit --load logs/my_run.log/best_model.pt \
        --fold 2 --shot 1 --reduce_dim 64 --sam2_ckpt ./pretrained/sam2_hiera_large.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.pascal5i import (
    build_loader, class_iou, denormalize, fold_class_ids, mean_iou_summary, to_device,
)
from models.fewshot import FewShotSeg
from utils import set_logger


def parse_args():
    parser = argparse.ArgumentParser('PASCAL-5i evaluation')
    parser.add_argument('--datapath', required=True)
    parser.add_argument('--benchmark', default='pascal', choices=['pascal', 'coco'])
    parser.add_argument('--load', required=True, help='checkpoint path')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--reduce_dim', type=int, default=64)
    parser.add_argument('--sam2_ckpt', default=None)
    parser.add_argument('--amp', type=int, default=1)
    parser.add_argument('--visualize', default=None, help='dir to dump prediction panels')
    parser.add_argument('--vis_n', type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp) and device.type == 'cuda'
    os.makedirs('logs', exist_ok=True)
    logger = set_logger(os.path.join('logs', 'test_pascal.log'))
    logger.info(f'args: {vars(args)}')

    loader = build_loader(
        args.datapath, fold=args.fold, img_size=args.img_size, split='val',
        shot=args.shot, bsz=args.bsz, nworker=args.nworker, benchmark=args.benchmark,
    )

    model = FewShotSeg(pretrained_weights=args.sam2_ckpt, reduce_dim=args.reduce_dim)
    state = torch.load(args.load, map_location='cpu')
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    ious = {c: [] for c in range(20)}
    vis_dir = args.visualize
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    episodes = min(args.n_episodes, len(loader.dataset))
    for idx, raw in enumerate(loader):
        if idx >= episodes:
            break
        batch = to_device(raw, device)
        query_img = batch['query_images'][0]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred, _, _ = model(
                batch['support_images'], batch['support_masks'],
                [query_img], train=False, return_all_stages=True,
            )

        # pick the refined final stage: (bsz, n_stages, 2, h, w)
        stages = pred.float().view(query_img.shape[0], -1, 2, *pred.shape[-2:])
        prob = stages[:, -1, 1:2]
        org_h, org_w = batch['org_size'][0].tolist()
        prob = F.interpolate(prob, size=(org_h, org_w), mode='bilinear', align_corners=True)
        pred_mask = (prob[0, 0] > 0.5)

        label = batch['query_labels'][0]
        ignore = batch['query_ignores'][0] > 0.5
        iou = class_iou(pred_mask, label, ignore=~ignore).item()
        ious[int(batch['class_id'][0])].append(iou)

        if vis_dir and idx < args.vis_n:
            _save_panel(os.path.join(vis_dir, f'{idx:04d}.png'), query_img[0], label, pred_mask)

    per_class, mean_over_classes = mean_iou_summary({c: v for c, v in ious.items() if v})
    all_fold_classes = fold_class_ids(args.fold)
    fold_ious = [per_class[c] for c in all_fold_classes if c in per_class]
    fold_mean = float(np.mean(fold_ious)) if fold_ious else float('nan')
    overall = float(np.mean([v for v in per_class.values() if not np.isnan(v)]))

    logger.info(f'per-class IoU: {per_class}')
    logger.info(f'fold {args.fold} ({args.shot}-shot) mIoU: {fold_mean * 100:.2f}')
    logger.info(f'mean IoU over evaluated classes: {overall * 100:.2f} '
                f'| mean over class means: {mean_over_classes * 100:.2f}')
    logger.info(f'episodes evaluated: {sum(len(v) for v in ious.values())}')
    return 0


def _save_panel(path, image, label, pred):
    import torchvision.utils as vutils

    h, w = image.shape[-2:]
    label = F.interpolate(label[None, None].float(), size=(h, w), mode='nearest')[0, 0]
    pred = F.interpolate(pred[None, None].float(), size=(h, w), mode='nearest')[0, 0]
    image = denormalize(image)
    overlay = torch.zeros_like(image)
    overlay[0] = pred * 0.6              # prediction in red
    overlay[1] = label * 0.6             # ground truth in green
    panel = torch.stack([image, (image + overlay).clamp(0, 1),
                         label.repeat(3, 1, 1), pred.repeat(3, 1, 1)], dim=0)
    vutils.save_image(panel, path, nrow=4, padding=2)


if __name__ == '__main__':
    raise SystemExit(main())
