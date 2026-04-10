import gc
import torch
import logging

from perseus.losses.focal import (
    FocalLoss
)
from perseus.models.initialize import build_optimizer
from perseus.losses.compute import compute_loss_from_batch
from perseus.trainer.evaluate import evaluate
from perseus.trainer.regularization import random_bin_masking_batch

logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, device, rank_idx_for_gate=None,
          epochs=10, lr=1e-3, save_path="model_cf.pt"):
    """
    Train a model using the provided training and validation DataLoaders

    Performs training for a specified number of epochs, evaluates on the validation set after each epoch,
    and saves the best model (by AUROC or lowest loss) to the specified path

    Args:
        model (torch.nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        device (torch.device): Device to run training on
        rank_idx_for_gate (int or None, optional): If set, applies a mask for the specified rank index. Defaults to None
        epochs (int, optional): Number of training epochs. Defaults to 10
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3
        save_path (str, optional): Path to save the best model weights. Defaults to "model_cf.pt"

    Returns:
        None
    """
    optim = build_optimizer(model, lr=lr, weight_decay=1e-4)
    crit  = FocalLoss(alpha=1, gamma=2)
    # crit = LineageAwareFocalLoss(
    #     gamma=2.0,
    #     alpha=0.25,
    #     lambda_hier=0.5,
    #     rank_weights=None 
    # )
    scaler = None
    # scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    torch.backends.cudnn.benchmark = True

    best_metric = -1.0
    best_state = None
    
    if hasattr(val_loader.batch_sampler, "set_epoch"):
        val_loader.batch_sampler.set_epoch(0) 

    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_n = 0
        
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(ep)

        for batch in train_loader:
            x   = batch["x"].to(device, non_blocking=True)
            msk = batch["mask"].to(device, non_blocking=True)
            extra = torch.log1p(batch["lengths"].to(device).float()).unsqueeze(1)
            
            # force tensors to fp32 on device
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            # y = y.to(device, non_blocking=True)
            if msk is not None:   msk = msk.to(device, non_blocking=True)           # bool/int ok
            if extra is not None: extra = extra.to(device, dtype=torch.float32, non_blocking=True)
            
            w = int((batch["labels_per_rank"] >= 0).sum().item())
            if w == 0: 
                continue

            # Apply random binary masking as data augmentation
            x = random_bin_masking_batch(x, msk, p=0.1)

            optim.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x, mask=msk, extra=extra)
                    loss = compute_loss_from_batch(logits, batch, device, crit, rank_idx_for_gate)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(x, mask=msk, extra=extra)
                loss = compute_loss_from_batch(logits, batch, device, crit, rank_idx_for_gate)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            total_loss += loss.item() * w
            total_n += w

        logger.info(f"Epoch {ep:02d} training complete, validating ...")
        train_loss = total_loss / max(total_n,1)

        val_metrics = evaluate(model, val_loader, device, rank_idx_for_gate)
        logger.info(f"Epoch {ep:02d} | train_loss={train_loss:.4f} | "
                 f"val_loss={val_metrics['loss']:.4f}" +
                 (f" | val_acc={val_metrics.get('acc',float('nan')):.4f} | val_auroc={val_metrics.get('auroc',float('nan')):.4f}"
                  if 'acc' in val_metrics else ""))

        score = -(val_metrics.get("loss", None))
        if score is None:
            score = -val_metrics["loss"]
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            logger.info(f"[saved best] {save_path}")
            
        torch.cuda.empty_cache(); gc.collect()

    if best_state is None:
        torch.save(model.state_dict(), save_path)
        logger.info(f"[saved last] {save_path}")