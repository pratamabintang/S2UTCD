r""" PASCAL-5i loaders for FewShotSeg episodes.

Wraps the episodic ``data.pascal.DatasetPASCAL`` sampler and reshapes each episode into
the contract expected by ``models.fewshot.FewShotSeg``:

    supp_imgs : list[n_way][n_shot] -> (bsz, 3, H, W)
    supp_mask : list[n_way][n_shot] -> (bsz, H, W)
    qry_imgs  : list[(bsz, 3, H, W)]
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import FSSDataset

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


class Pascal5iEpisodes(Dataset):
    """Per-episode view of PASCAL-5i (query + support + binary mask of the sampled class)."""

    def __init__(self, datapath, fold, img_size, split='trn', shot=1,
                 benchmark='pascal', use_original_imgsize=None):
        FSSDataset.initialize(img_size=img_size, datapath=datapath)
        dataset_cls = FSSDataset.datasets[benchmark]
        if use_original_imgsize is None:
            # keep masks at native resolution for evaluation, resize for training
            use_original_imgsize = split in ('val', 'test')
        self.shot = shot
        self.split = split
        self.inner = dataset_cls(
            datapath,
            fold=fold,
            transform=FSSDataset.transform,
            split=split,
            shot=shot,
            use_original_imgsize=use_original_imgsize,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        episode = self.inner[idx]
        support_images = [[episode['support_imgs'][s] for s in range(self.shot)]]
        support_masks = [[episode['support_masks'][s] for s in range(self.shot)]]
        org_w, org_h = episode['org_query_imsize']
        return {
            'support_images': support_images,
            'support_masks': support_masks,
            'query_images': [episode['query_img']],
            'query_labels': episode['query_mask'],
            'query_ignores': episode['query_ignore_idx'],
            'class_id': int(episode['class_id']),
            'query_name': episode['query_name'],
            'org_size': torch.tensor([org_h, org_w]),  # (H, W)
        }


def collate_episodes(batch):
    """Stack ``bsz`` episodes while keeping the n_way / n_shot list layout."""
    n_way = len(batch[0]['support_images'])
    n_shot = len(batch[0]['support_images'][0])

    support_images = [
        [torch.stack([ep['support_images'][w][s] for ep in batch], dim=0) for s in range(n_shot)]
        for w in range(n_way)
    ]
    support_masks = [
        [torch.stack([ep['support_masks'][w][s] for ep in batch], dim=0) for s in range(n_shot)]
        for w in range(n_way)
    ]
    return {
        'support_images': support_images,
        'support_masks': support_masks,
        'query_images': [torch.stack([ep['query_images'][0] for ep in batch], dim=0)],
        'query_labels': torch.stack([ep['query_labels'] for ep in batch], dim=0),
        'query_ignores': torch.stack([ep['query_ignores'] for ep in batch], dim=0),
        'class_id': torch.tensor([ep['class_id'] for ep in batch]),
        'query_name': [ep['query_name'] for ep in batch],
        'org_size': torch.stack([ep['org_size'] for ep in batch], dim=0),
    }


def build_loader(datapath, fold, img_size, split='trn', shot=1, bsz=1, nworker=0,
                 benchmark='pascal'):
    dataset = Pascal5iEpisodes(datapath, fold=fold, img_size=img_size, split=split,
                               shot=shot, benchmark=benchmark)
    return DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=split == 'trn',
        num_workers=nworker if split == 'trn' else 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == 'trn',
        collate_fn=collate_episodes,
    )


def to_device(batch, device):
    """Move an episode batch produced by ``collate_episodes`` to ``device``."""
    batch['support_images'] = [
        [t.float().to(device) for t in way] for way in batch['support_images']
    ]
    batch['support_masks'] = [
        [t.float().to(device) for t in way] for way in batch['support_masks']
    ]
    batch['query_images'] = [t.float().to(device) for t in batch['query_images']]
    batch['query_labels'] = batch['query_labels'].float().to(device)
    batch['query_ignores'] = batch['query_ignores'].float().to(device)
    batch['class_id'] = batch['class_id'].to(device)
    return batch


def denormalize(img, device=None):
    """Undo ImageNet normalization for visualization; img: (3, H, W)."""
    device = device or img.device
    mean = torch.tensor(IMG_MEAN, device=device, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(IMG_STD, device=device, dtype=img.dtype).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def class_iou(pred, label, ignore=None, eps=1e-8):
    """IoU of a binary prediction, optionally ignoring boundary/void pixels."""
    pred = pred.bool()
    label = label.bool()
    valid = torch.ones_like(label, dtype=torch.bool) if ignore is None else ignore.bool()
    tp = (pred & label & valid).sum().float()
    fp = (pred & ~label & valid).sum().float()
    fn = (~pred & label & valid).sum().float()
    return tp / (tp + fp + fn + eps)


def fold_class_ids(fold, nfolds=4, nclass=20):
    nclass_trn = nclass // nfolds
    return list(range(fold * nclass_trn, fold * nclass_trn + nclass_trn))


def mean_iou_summary(ious_per_class):
    """Per-class mean IoU plus the PASCAL-5i average (mean over classes)."""
    per_class = {c: float(np.mean(v)) if len(v) else float('nan') for c, v in ious_per_class.items()}
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return per_class, float(np.mean(valid)) if valid else float('nan')
