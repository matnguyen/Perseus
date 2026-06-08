CANONICAL_RANKS = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
RANK_INDEX = {r: i for i, r in enumerate(CANONICAL_RANKS)}
NUM_RANKS = len(CANONICAL_RANKS)
N_CHANNELS = 22
CROP_MAX_T = 4096
DEFAULT_MODEL_FILE = "trained_weights.pt"

# Ablation: all permutations of (in_lineage, out_of_lineage, descendant)
CHANNEL_ORDERS = {
    "fi_fo_fd": (0, 1, 2),  # default
    "fi_fd_fo": (0, 2, 1),
    "fo_fi_fd": (1, 0, 2),
    "fo_fd_fi": (1, 2, 0),
    "fd_fi_fo": (2, 0, 1),
    "fd_fo_fi": (2, 1, 0),
}
DEFAULT_CHANNEL_ORDER = "fi_fo_fd"