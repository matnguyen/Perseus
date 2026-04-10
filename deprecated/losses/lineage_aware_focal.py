class LineageAwareFocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        lambda_hier: float = 0.5,
        rank_weights=None,
    ):
        """
        Lineage-aware focal loss for canonical ranks ordered GENERAL → SPECIFIC.

        logits:  [B, R]
        targets: [B, R] in {0, 1, -1}, where -1 means "unknown / ignore"
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_hier = lambda_hier

        if rank_weights is not None:
            self.register_buffer(
                "rank_weights",
                torch.as_tensor(rank_weights, dtype=torch.float32)
            )
        else:
            self.rank_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, R] raw logits
            targets: [B, R] in {0,1,-1}; -1 means "ignore"
            mask: optional mask [B, R] (0/1 or bool) to additionally ignore elements

        Returns:
            Scalar loss (focal + lambda_hier * hierarchy_penalty), averaged over valid elements.
        """
        device = logits.device
        B, R = logits.shape

        # ----- VALID MASK -----
        # Start from "targets >= 0" to ignore -1s, then combine with external mask if given.
        valid_mask = (targets >= 0)
        if mask is not None:
            # allow float mask; treat >0 as True
            valid_mask = valid_mask & (mask > 0)

        # Clamp targets to [0,1] for BCE/focal
        y = targets.clamp(min=0).float()

        # ----- FOCAL LOSS -----
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p   = torch.sigmoid(logits)
        p_t = torch.where(y > 0.5, p, 1 - p)

        if self.alpha is not None:
            alpha_t = torch.where(
                y > 0.5,
                torch.tensor(self.alpha, device=device),
                torch.tensor(1 - self.alpha, device=device),
            )
        else:
            alpha_t = torch.ones_like(p_t)

        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        focal_elementwise = focal_weight * bce

        if self.rank_weights is not None:
            rw = self.rank_weights.view(1, R).expand_as(focal_elementwise)
            focal_elementwise = focal_elementwise * rw

        num_valid = valid_mask.sum()
        if num_valid > 0:
            focal_loss = focal_elementwise[valid_mask].mean()
        else:
            focal_loss = torch.tensor(0.0, device=device)

        # ----- HIERARCHY PENALTY -----
        # Canonical order: parent (general) at index 0 → child (specific) at index R-1
        parent_probs = p[:, :-1]   # general
        child_probs  = p[:, 1:]    # specific

        # A violation: child > parent
        violations = torch.relu(child_probs - parent_probs)

        # Valid rank-pairs must have both parent & child valid
        valid_pairs = valid_mask[:, :-1] & valid_mask[:, 1:]

        if valid_pairs.any():
            hier_penalty = (violations[valid_pairs] ** 2).mean()
        else:
            hier_penalty = torch.tensor(0.0, device=device)

        return focal_loss + self.lambda_hier * hier_penalty

