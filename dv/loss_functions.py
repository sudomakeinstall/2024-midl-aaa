# System
import typing as ty

# Third Party
import torch as t


class JaccardLoss(t.nn.Module):
    """A module for Jaccard (intersection-over-union) loss."""

    def __init__(
        self,
        epsilon: float = 10e-7,
        include_background: bool = False,
        weights: ty.Optional[t.Tensor] = None,
    ):
        """
        Initialization.

        Args:
            epsilon: Small factor in the IOU denominator to prevent numerical error.
            include_background: If `False`, the background (`0`) is excluded.
            weights: A weighting factor for each class.
        """
        super(JaccardLoss, self).__init__()
        self.epsilon = epsilon
        self.include_background = include_background
        if weights is not None:
            assert (
                weights.dim() == 2
            ), "Weights should have batch and class dimensions only."
        self.weights = weights

    def forward(self, seg_pd, seg_gt):
        seg_pd = t.nn.functional.softmax(seg_pd, dim=1)

        if not self.include_background:
            seg_pd = seg_pd[:, 1:]
            seg_gt = seg_gt[:, 1:]

        N = seg_pd.size()[0]
        C = seg_pd.size()[1]

        assert seg_gt.size()[0] == N, "Unequal batch size."
        assert seg_gt.size()[1] == C, "Unequal channel size."

        if self.weights is None:
            self.weights = t.ones((1, C), device=seg_pd.device) / C

        intersection = (seg_pd * seg_gt).view(N, C, -1).sum(2)
        cardinality = (seg_pd + seg_gt).view(N, C, -1).sum(2)
        union = cardinality - intersection
        iou = intersection / union.clamp_min(self.epsilon)
        iou_weighted = iou * self.weights

        return 1 - iou_weighted.sum(1)
