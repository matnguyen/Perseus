from importlib import resources
import torch
import logging

from perseus.models.cnn import CNN1D_CF
from perseus.utils.constants import N_CHANNELS
from perseus.utils.constants import DEFAULT_MODEL_FILE

LOG = logging.getLogger(__name__)

def make_model(out_dim, device):
    """
    Instantiate and return the selected model architecture.

    Args:
        out_dim (int): Output dimension for the model (1 for binary, 7 for per-rank).

    Returns:
        torch.nn.Module: Instantiated model moved to the selected device.
    """
    LOG.info("Model: CNN1D_CF (out_dim=%d)", out_dim)
    return CNN1D_CF(in_channels=N_CHANNELS, out_dim=out_dim, extra_dim=1).to(device)

def load_default_model(out_dim, device):
    """
    Load the default model architecture with pretrained weights.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model with pretrained weights.
    """
    model = make_model(out_dim, device)
    
    # Locate packaged model file
    model_file = resources.files("perseus.models") / DEFAULT_MODEL_FILE
    
    with resources.as_file(model_file) as checkpoint_path:
        return load_model(model, checkpoint_path, device)
    

def load_model(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
    """
    LOG.info("Loading model weights from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)

def build_optimizer(model, lr=1e-3, weight_decay=1e-4):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # DO NOT apply weight decay to:
        #   - biases   (name.endswith("bias"))
        #   - BatchNorm / LayerNorm parameters (1D)
        if param.ndim == 1 or name.endswith("bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer