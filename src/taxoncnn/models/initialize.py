import torch
import logging

from taxoncnn.models.cnn import CNN1D_CF
from taxoncnn.models.restcn import ResTCN_CF
from taxoncnn.utils.constants import N_CHANNELS

LOG = logging.getLogger(__name__)

def make_model(args, out_dim, device):
    """
    Instantiate and return the selected model architecture.

    Args:
        out_dim (int): Output dimension for the model (1 for binary, 7 for per-rank).

    Returns:
        torch.nn.Module: Instantiated model moved to the selected device.
    """
    if args.model == "cnn":
        LOG.info("Model: CNN1D_CF (out_dim=%d)", out_dim)
        return CNN1D_CF(in_channels=N_CHANNELS, out_dim=out_dim, extra_dim=1).to(device)
    else:
        LOG.info("Model: ResTCN_CF (out_dim=%d)", out_dim)
        return ResTCN_CF(in_channels=N_CHANNELS, out_dim=out_dim, extra_dim=1).to(device)
    

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