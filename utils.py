import torch
import torch.distributed as dist
import torch.nn.functional as F

def compute_metrics(flow_pred, flow_gt, valid_flow_mask=None):

    epe = ((flow_pred - flow_gt) ** 2).sum(dim=1).sqrt()
    flow_norm = (flow_gt ** 2).sum(dim=1).sqrt()

    if valid_flow_mask is not None:
        epe = epe[valid_flow_mask]
        flow_norm = flow_norm[valid_flow_mask]

    relative_epe = epe / flow_norm

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
        "f1": ((epe > 3) & (relative_epe > 0.05)).float().mean().item() * 100,
    }
    return metrics, epe.numel()
