#!/usr/bin/env python
r"""Train FewShotSeg (SAM2 encoder, FEBR/UCI decoder) on PASCAL-5i.

Example:
    python train_pascal.py --datapath ./data/VOCdevkit --benchmark pascal \
        --logpath log_pascal_fold2_1shot --fold 2 --shot 1 --bsz 4 \
        --sam2_ckpt ./pretrained/sam2_hiera_large.pt --niter 20000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.pascal5i import build_loader, to_device
from models.fewshot import FewShotSeg
from utils import set_logger


def parse_args():
    parser = argparse.ArgumentParser('PASCAL-5i training')
    parser.add_argument('--datapath', required=True, help='folder that contains VOC2012/')
    parser.add_argument('--benchmark', default='pascal', choices=['pascal', 'coco'])
    parser.add_argument('--logpath', required=True, help='run name; checkpoints go to logs/<logpath>.log/')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--niter', type=int, default=20000)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--reduce_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adapter_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--sam2_ckpt', default=None, help='path to sam2_hiera_*.pt')
    parser.add_argument('--amp', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=2000)
    return parser.parse_args()


def build_optimizer(model, args):
    adapter, rest = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (adapter if 'prompt_learn' in name else rest).append(param)
    groups = [{'params': rest, 'lr': args.lr}]
    if adapter:
        groups.append({'params': adapter, 'lr': args.adapter_lr})
    return torch.optim.SGD(groups, momentum=args.momentum, weight_decay=args.weight_decay)


def masked_bce(fg_prob, target, weight, eps=1e-6):
    prob = fg_prob.clamp(eps, 1.0 - eps)
    loss = F.binary_cross_entropy(prob, target, reduction='none')
    return (loss * weight).sum() / (weight.sum() + eps)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(args.amp) and device.type == 'cuda'
    if device.type != 'cuda':
        print('WARNING: CUDA not found, running on CPU (very slow, for smoke tests only)')

    run_dir = os.path.join('logs', f'{args.logpath}.log')
    os.makedirs(run_dir, exist_ok=True)
    logger = set_logger(os.path.join(run_dir, 'logger.log'))
    logger.info(f'args: {vars(args)}')

    loader = build_loader(
        args.datapath, fold=args.fold, img_size=args.img_size, split='trn',
        shot=args.shot, bsz=args.bsz, nworker=args.nworker, benchmark=args.benchmark,
    )
    logger.info(f'train episodes: {len(loader.dataset)} | iterations: {args.niter}')

    model = FewShotSeg(pretrained_weights=args.sam2_ckpt, reduce_dim=args.reduce_dim).to(device)
    model.train()
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.niter, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    it = iter(loader)
    running = {'total': 0.0, 'seg': 0.0, 'rec': 0.0, 'align': 0.0}
    best = float('inf')
    start = time.time()

    for step in range(1, args.niter + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = to_device(batch, device)

        query_img = batch['query_images'][0]
        query_labels = batch['query_labels']                       # (bsz, H, W)
        query_ignore = batch['query_ignores']                      # (bsz, H, W)
        bsz = query_img.shape[0]
        target = query_labels.unsqueeze(1)                         # (bsz, 1, H, W)
        weight = 1.0 - query_ignore.unsqueeze(1)                   # drop boundary / void pixels

        # masked query used as the reconstruction target (fg region removed)
        masked = query_img[:, 0:1]
        fg = target.bool()
        fill = masked.amin(dim=(1, 2, 3), keepdim=True)
        masked = torch.where(fg, fill.expand_as(masked), masked)
        query_images = [query_img, masked.repeat(1, 3, 1, 1)]
        remove_mask = target

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            query_pred, rec_out, align_loss = model(
                batch['support_images'], batch['support_masks'], query_images, train=True
            )
            stages = query_pred.view(bsz, -1, 2, *query_pred.shape[-2:])
            seg_loss = sum(
                masked_bce(stages[:, s, 1:2].float(), target, weight)
                for s in range(stages.shape[1])
            ) / stages.shape[1]
            rec_err = (masked - rec_out.float()).pow(2).mean(dim=1, keepdim=True)
            rec_loss = (rec_err * remove_mask).sum() / (remove_mask.sum() + 1e-6)
            loss = seg_loss + rec_loss + align_loss.float()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        last_loss = loss.item()
        running['total'] += last_loss
        running['seg'] += seg_loss.item()
        running['rec'] += rec_loss.item()
        running['align'] += align_loss.item()

        if step % args.print_every == 0:
            n = args.print_every
            elapsed = (time.time() - start) / n
            logger.info(
                f'step {step}/{args.niter} | total {running["total"] / n:.4f} '
                f'| seg {running["seg"] / n:.4f} | rec {running["rec"] / n:.4f} '
                f'| align {running["align"] / n:.4f} | {elapsed:.2f}s/it'
            )
            running = {k: 0.0 for k in running}
            start = time.time()

        if step % args.save_every == 0 or step == args.niter:
            torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pt'))
            if last_loss < best:
                best = last_loss
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pt'))
            logger.info(f'snapshot saved at step {step} (loss {last_loss:.4f}, best {best:.4f})')

    logger.info('training done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
