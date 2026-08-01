# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss, weighted_loss
from mmcv.ops.diff_iou_rotated import oriented_box_intersection_2d
from torch import Tensor
from torch.nn import functional as F
import torch 


@weighted_loss
def bce_with_logits_loss(pred: Tensor, target: Tensor, pos_weight: Optional[Tensor] = None) -> Tensor:
    """ """

    losses = F.binary_cross_entropy_with_logits(
        pred,
        target,
        None,  # Always None since the weight will be used in the weighted_loss wrapper
        pos_weight=pos_weight,
        reduction="none",  # Always none since the reduction will happen in the weighted_loss wrapper
    )
    return losses


@weighted_loss
def corner_l1_losses(pred: Tensor, target: Tensor, norm_values: Optional[Tensor] = None, barrier_masks=None) -> Tensor:
    """ """

    # Shape: [B, N, 4, 2] -> [B, N, 4] -> [B, N]
    direct_losses = (pred - target).abs().sum(dim=-1).mean(dim=-1)

    losses = direct_losses
    # Branch out barrier if set
    if barrier_masks is not None:

        # [c0, c1, c2, c3] -> [c2, c3, c0, c1]
        target_shifted_180 = torch.roll(
            target,
            shifts=2,
            dims=-2,
        )

        shifted_losses = (pred - target_shifted_180).abs().sum(dim=-1).mean(dim=-1)  # [B, N]

        barrier_losses = torch.minimum(
            direct_losses,
            shifted_losses,
        )

        # Use symmetry-aware loss only for barriers.
        losses = torch.where(
            barrier_masks,
            barrier_losses,
            direct_losses,
        )

    if norm_values is not None:
        losses = losses / norm_values
    return losses


@weighted_loss
def giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """ """

    losses = target - pred 
    return losses


@MODELS.register_module()
class CustomBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss"""

    def __init__(self, loss_weight=None, reduction="mean", pos_weight=None) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[Union[int, float]] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        """Forward function.

        If you want to manually determine which positions are
        positive samples, you can set the pos_index and pos_label
        parameter. Currently, only the CenterNet update version uses
        the parameter.

        Args:
            pred (torch.Tensor): The prediction. The shape is (N, num_classes).
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution. The shape is (N, num_classes).
            pos_inds (torch.Tensor): The positive sample index.
                Defaults to None.
            pos_labels (torch.Tensor): The label corresponding to the positive
                sample index. Defaults to None.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, float, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        losses = self.loss_weight * bce_with_logits_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor, pos_weight=self.pos_weight
        )
        return losses


@MODELS.register_module()
class BEVCornerLoss(nn.Module):
    """Compute 4 corner losses between predictions and gt boxes."""

    def __init__(
        self,
        out_size_factor,
        voxel_size,
        pc_range,
        gt_diagonal_norm: bool,
        cone_label_index: Optional[int],
        barrier_label_index: Optional[int],
        loss_weight=1.0,
        reduction="mean",
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.gt_diagonal_norm = gt_diagonal_norm
        self.cone_label_index = cone_label_index
        self.barrier_label_index = barrier_label_index

    def _convert_to_bev_corners(
        self, 
        bboxes: Tensor, 
        labels: Tensor, 
        orientations: Tensor,
        is_gt: bool = False) -> Tensor:
        """
        bboxes (B, num_proposal, 10)
        """
        batch_size = bboxes.shape[0]
        center_x = bboxes[:, :, 0] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center_y = bboxes[:, :, 1] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        lw = bboxes[:, :, 3:5].exp()
        if orientations is not None:
            rot_sin = orientations[:, :, 0:1]
            rot_cos = orientations[:, :, 1:2]
        else:
            rot_sin = bboxes[:, :, 6:7]
            rot_cos = bboxes[:, :, 7:8]
        
        if not is_gt:
            norm_rotation = torch.sqrt(rot_sin.square() + rot_cos.square() + 1e-6)
            rot_sin = rot_sin / norm_rotation
            rot_cos = rot_cos / norm_rotation

        row1 = torch.cat([rot_cos, rot_sin], dim=-1)
        row2 = torch.cat([-rot_sin, rot_cos], dim=-1)  # (B, N, 2)
        rotation_matrix_transpose = torch.stack([row1, row2], dim=-2)  # (B, N, 2, 2)

        if self.cone_label_index is not None:
            cone_mask = (labels == self.cone_label_index)[..., None, None]  # [B, N, 1, 1]

            identity = torch.eye(
                2,
                device=rotation_matrix_transpose.device,
                dtype=rotation_matrix_transpose.dtype,
            ).view(1, 1, 2, 2)

            rotation_matrix_transpose = torch.where(
                cone_mask,
                identity,
                rotation_matrix_transpose,
            )

        x4 = lw.new_tensor([0.5, -0.5, -0.5, 0.5]).to(lw.device)
        x4 = x4 * lw[:, :, 0].unsqueeze(-1)  # (B, N, 4)
        y4 = lw.new_tensor([0.5, 0.5, -0.5, -0.5]).to(lw.device)
        y4 = y4 * lw[:, :, 1].unsqueeze(-1)  # (B, N, 4)
        # (top right, top left, bottom left, bottom right)
        corners = torch.stack([x4, y4], dim=-1)  # (B, N, 4, 2)

        # (B * N, 4, 2) @ (B * N, 2, 2) -> (B * N, 4, 2)
        rotated = torch.bmm(corners.view([-1, 4, 2]), rotation_matrix_transpose.view([-1, 2, 2]))
        rotated = rotated.view([batch_size, -1, 4, 2])  # (B * N, 4, 2) -> (B, N, 4, 2)
        # Translation
        rotated[..., 0] += center_x.unsqueeze(-1)
        rotated[..., 1] += center_y.unsqueeze(-1)

        # Diagonal values, (B, N,)
        # lw_diagonal = torch.sqrt(lw[:, :, 0].square() + lw[:, :, 1].square() + 1e-6)
        return rotated, lw

    def forward(
        self,
        preds_bboxes: Tensor,
        gts_bboxes: Tensor,
        labels: Tensor,
        weight: Tensor,
        avg_factor: Optional[Union[int, float]] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        """
        preds_bboxes (B, num_proposals, 10)
        gts_bboxes (B, num_proposals, 10)
        labels (B, num_proposals, )
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        preds_corners, _ = self._convert_to_bev_corners(bboxes=preds_bboxes, labels=labels)
        gts_corners, gts_length_width = self._convert_to_bev_corners(bboxes=gts_bboxes, labels=labels)
        if self.gt_diagonal_norm:
            norm_values = torch.sqrt(gts_length_width[:, :, 0].square() + gts_length_width[:, :, 1].square() + 1e-6)
        else:
            norm_values = None

        if self.barrier_label_index is not None:
            barrier_masks = labels == self.barrier_label_index
        else:
            barrier_masks = None

        losses = corner_l1_losses(
            preds_corners,
            gts_corners,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            norm_values=norm_values,
            barrier_masks=barrier_masks,
        )  # (B, N)

        return self.loss_weight * losses


@MODELS.register_module()
class RotatedBEVGIOULoss(BEVCornerLoss):
    """Compute rotated GIOU loss between predictions and gt boxes."""

    def __init__(
        self,
        out_size_factor,
        voxel_size,
        pc_range,
        gt_diagonal_norm: bool,
        cone_label_index: Optional[int],
        barrier_label_index: Optional[int],
        giou: bool=True,
        loss_weight=1.0,
        reduction="mean",
    ) -> None:
        super().__init__(
            out_size_factor=out_size_factor,
            voxel_size=voxel_size,
            pc_range=pc_range,
            gt_diagonal_norm=gt_diagonal_norm,
            cone_label_index=cone_label_index,
            barrier_label_index=barrier_label_index,
            loss_weight=loss_weight,
            reduction=reduction,
        )
        self.giou = giou
    
    def convex_hull_area(self, corners1: Tensor, corners2: Tensor) -> Tensor:
        """Area of the convex hull of two rotated boxes.

        The tightest convex region containing both boxes, i.e. the smallest
        enclosing convex object of the original GIoU formulation.

        Note:
                A corner with minimal x always lies on the hull, and the hull is
                star-shaped with respect to it, so the hull vertices are met in
                increasing angular order around it. The convex hull is also the
                maximum-area polygon on the point set, so its area is the largest fan
                area over all sub-sequences of that angular order, which a small
                dynamic program finds exactly. Ties, duplicated and collinear corners
                need no special casing because a chain through an interior corner is
                never the maximum.

        Args:
                corners1 (Tensor): (B, N, 4, 2) First batch of boxes.
                corners2 (Tensor): (B, N, 4, 2) Second batch of boxes.

        Returns:
                Tensor: (B, N) Area of the convex hull.
        """
        corners = torch.cat([corners1, corners2], dim=2)  # (B, N, 8, 2)
        num = corners.size(2)
        anchor_idx = corners[..., 0].argmin(dim=2)  # (B, N)
        anchor_idx = anchor_idx[..., None, None].expand(-1, -1, 1, 2)
        # corners relative to the anchor, (B, N, 8, 2)
        rel = corners - torch.gather(corners, 2, anchor_idx)
        # all corners lie in the half plane x >= 0, so the angles fall in
        # [-pi / 2, pi / 2] and sorting them needs no branch cut handling
        order = torch.atan2(rel[..., 1], rel[..., 0]).argsort(dim=2)
        rel = torch.gather(rel, 2, order.unsqueeze(-1).expand(-1, -1, -1, 2))
        x, y = rel.split([1, 1], dim=-1)
        # cross[..., i, j] is twice the area of the triangle (anchor, i, j) and is
        # non-negative for i < j, (B, N, 8, 8)
        cross = x * y.transpose(-1, -2) - y * x.transpose(-1, -2)
        # best[j]: largest fan area of a chain ending at the j-th corner
        best = [torch.zeros_like(cross[..., 0, 0])]
        for j in range(1, num):
            best.append(
                (torch.stack(best, dim=-1) + cross[..., :j, j]).max(dim=-1)[0])
        return torch.stack(best, dim=-1).max(dim=-1)[0] / 2

    def forward(
        self,
        preds_bboxes: Tensor,
        gts_bboxes: Tensor,
        preds_orientation: Tensor,
        gts_orientation: Tensor,
        labels: Tensor,
        weight: Tensor,
        avg_factor: Optional[Union[int, float]] = None,
        reduction_override: Optional[str] = None,
    ) -> Tensor:
        """
        preds_bboxes (B, num_proposals, 10)
        gts_bboxes (B, num_proposals, 10)
        labels (B, num_proposals, )
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        preds_corners, preds_length_width = self._convert_to_bev_corners(bboxes=preds_bboxes, orientations=preds_orientation, labels=labels, is_gt=False)
        gts_corners, gts_length_width = self._convert_to_bev_corners(bboxes=gts_bboxes, orientations=gts_orientation, labels=labels, is_gt=True)
        
        intersection, _ = oriented_box_intersection_2d(preds_corners, gts_corners)  # (B, N)    
        area1 = preds_length_width[:, :, 0] * preds_length_width[:, :, 1]
        area2 = gts_length_width[:, :, 0] * gts_length_width[:, :, 1]
        union = area1 + area2 - intersection
        iou = intersection / union
        
        if self.giou:
            convex_hull_area = self.convex_hull_area(preds_corners, gts_corners)
            giou = iou - (convex_hull_area - union) / convex_hull_area
        else:
            giou = iou
            
        targets = torch.ones_like(giou)

        losses = giou_loss(
            giou, 
            targets,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * losses
    