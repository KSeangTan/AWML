# modify from https://github.com/mit-han-lab/bevfusion
import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner, BaseBBoxCoder

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from mmdet3d.registry import TASK_UTILS
from mmengine.structures import InstanceData


def limit_period(val: torch.Tensor,
                 offset: float = 0.5,
                 period: float = 2 * torch.pi) -> torch.Tensor:
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to torch.pi.

    Returns:
        torch.Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


@TASK_UTILS.register_module()
class TransFusionBBoxCoder(BaseBBoxCoder):

    def __init__(
        self,
        pc_range,
        out_size_factor,
        voxel_size,
        num_orientation_bins=4,
        orientation_bin_offset=torch.pi / 4,    
        circle_orientation=True,
        post_center_range=None,
        score_threshold=None,
        code_size=8,
    ):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.circle_orientation = circle_orientation
        self.code_size = code_size
        self.num_orientation_bins = num_orientation_bins
        self.orientation_bin_offset = orientation_bin_offset 
        self.bin_size = 2 * torch.pi / self.num_orientation_bins

    def encode(self, dst_boxes):
        targets = torch.zeros([dst_boxes.shape[0], self.code_size]).to(dst_boxes.device)
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        # bottom center to gravity center
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        direction_targets = None
        if not self.circle_orientation:
            yaws = dst_boxes[:, 6].clone()
            # Fold the yaw into [0, 2 * pi) once, then derive BOTH the bin index and the
            # residual from that same folded value. Deriving the residual from the raw
            # `yaws` instead leaves a 2 * pi offset on every box whose yaw falls below
            # `orientation_bin_offset`, which puts the regression target near -2 * pi
            # instead of inside [-bin_size / 2, bin_size / 2).
            shifted_yaws = limit_period(yaws - self.orientation_bin_offset, offset=0.0, period=2 * torch.pi)
            # values in [0, num_orientation_bins - 1]; clamp guards the case where
            # `shifted_yaws` lands on 2 * pi through floating point rounding.
            direction_targets = torch.floor(shifted_yaws / self.bin_size).long().clamp(
                min=0, max=self.num_orientation_bins - 1
            )
            # value in [-bin_size / 2, bin_size / 2), i.e. [-pi/4, pi/4) for 4 bins
            residual_yaws = shifted_yaws - (direction_targets.float() * self.bin_size + self.bin_size / 2)
            targets[:, 6] = residual_yaws
        else:
            targets[:, 6] = torch.sin(dst_boxes[:, 6])
            targets[:, 7] = torch.cos(dst_boxes[:, 6])
        
        if self.code_size == 9:
            targets[:, 7:9] = dst_boxes[:, 7:]
        elif self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
                
        # if not self.circle_orientation:
        #     # Get the direction, where 0 means front and 1 means back. The direction is used to determine the orientation of the box.
        #     is_front = (dst_boxes[:, 6] > -torch.pi / 2) & (dst_boxes[:, 6] <= torch.pi / 2)
        #     direction_targets[:, 0] = (~is_front).to(direction_targets.dtype)
        return targets, direction_targets.unsqueeze(-1) if direction_targets is not None else None

    def decode(self, heatmap, rot, dim, center, height, vel, directions, filter=False):
        """Decode bboxes.
        Args:
            heat (torch.Tensor): Heatmap with the shape of
                [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with the shape of
                [B, 1, num_proposals].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            height (torch.Tensor): height of the boxes with the shape of
                [B, 2, num_proposals]. (in real world metric)
            vel (torch.Tensor): Velocity with the shape of
                [B, 2, num_proposals].
            directions (torch.Tensor): Direction with the shape of [B, 1, num_proposals]
            filter: if False, return all box without checking score and
                center_range
        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5  # gravity center to bottom center
        if not self.circle_orientation:
            yaw_pred = rot[:, 0:1, :]  # [BS, 1, num_proposals]
            # keepdim so this stays [BS, 1, num_proposals]. Without it the index is
            # [BS, num_proposals], which broadcasts against yaw_pred to
            # [BS, BS, num_proposals] -- pairing every sample's residual with every
            # other sample's bin, and widening the decoded box tensor so that the
            # velocity columns shift out of place.
            final_directions = directions.argmax(dim=1, keepdim=True)
            # The residual is only meaningful inside its own bin, so clamp before
            # un-binning. This also keeps a residual head that has drifted (the
            # sin() training objective is satisfied at both r and r + pi) from
            # producing a full half-turn error.
            yaw_pred = yaw_pred.clamp(min=-self.bin_size / 2, max=self.bin_size / 2)
            # Direct inverse of encode(). Wrapping the residual with limit_period over
            # bin_size instead would snap any prediction that drifts outside
            # [-bin_size / 2, bin_size / 2) into the neighbouring bin, turning a small
            # regression error into a discontinuous bin_size jump.
            rot = yaw_pred + self.orientation_bin_offset + (
                final_directions.float() * self.bin_size + self.bin_size / 2
            )
            rot = limit_period(rot, offset=0.5, period=2 * torch.pi)  # limit to [-pi, pi)
        else:
            rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
            rot = torch.atan2(rots, rotc)
        
        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        if not filter:
            for i in range(heatmap.shape[0]):
                boxes3d = final_box_preds[i]
                scores = final_scores[i]
                labels = final_preds[i]
                predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}
                predictions_dicts.append(predictions_dict)
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            if isinstance(self.score_threshold, float):
                thresh_mask = final_scores > self.score_threshold
            elif isinstance(self.score_threshold, (list, tuple)):
                score_threshold = final_scores.new_tensor(self.score_threshold)
                thresh_mask = final_scores > score_threshold[final_preds]
            else:
                raise ValueError("score_threshold must be a float or list")

        predictions_dicts = []
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=heatmap.device)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)

            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}
                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )

        return predictions_dicts


@TASK_UTILS.register_module()
class BBoxBEVL1Cost(object):

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        pc_start = bboxes.new(train_cfg["point_cloud_range"][0:2])
        pc_range = bboxes.new(train_cfg["point_cloud_range"][3:5]) - bboxes.new(train_cfg["point_cloud_range"][0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost(object):

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = -iou
        return iou_cost * self.weight


@TASK_UTILS.register_module()
class HeuristicAssigner3D(BaseAssigner):

    def __init__(self, dist_thre=100, iou_calculator=dict(type="BboxOverlaps3D")):
        self.dist_thre = dist_thre  # distance in meter
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, query_labels=None):
        dist_thre = self.dist_thre
        num_gts, num_bboxes = len(gt_bboxes), len(bboxes)

        bev_dist = torch.norm(
            bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :], dim=-1
        )  # [num_gts, num_bboxes]
        if query_labels is not None:
            # only match the gt box and query with same category
            not_same_class = query_labels[None] != gt_labels[:, None]
            bev_dist += not_same_class * dist_thre

        # for each gt box, assign it to the nearest pred box
        nearest_values, nearest_indices = bev_dist.min(1)  # [num_gts]
        assigned_gt_inds = (
            torch.ones(
                [
                    num_bboxes,
                ]
            ).to(bboxes)
            * 0
        )
        assigned_gt_vals = (
            torch.ones(
                [
                    num_bboxes,
                ]
            ).to(bboxes)
            * 10000
        )
        assigned_gt_labels = (
            torch.ones(
                [
                    num_bboxes,
                ]
            ).to(bboxes)
            * -1
        )
        for idx_gts in range(num_gts):
            # for idx_pred in torch.where(bev_dist[idx_gts] < dist_thre)[0]:
            # # each gt match to all the pred box within some radius
            idx_pred = nearest_indices[idx_gts]  # each gt only match to the nearest pred box
            if bev_dist[idx_gts, idx_pred] <= dist_thre:
                # if this pred box is assigned, then compare
                if bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]:
                    assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
                    # for AssignResult, 0 is negative, -1 is ignore, 1-based
                    # indices are positive
                    assigned_gt_inds[idx_pred] = idx_gts + 1
                    assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

        max_overlaps = torch.zeros(
            [
                num_bboxes,
            ]
        ).to(bboxes)
        matched_indices = torch.where(assigned_gt_inds > 0)
        matched_iou = self.iou_calculator(
            gt_bboxes[assigned_gt_inds[matched_indices].long() - 1], bboxes[matched_indices]
        ).diag()
        max_overlaps[matched_indices] = matched_iou

        return AssignResult(num_gts, assigned_gt_inds.long(), max_overlaps, labels=assigned_gt_labels)


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):

    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),
        reg_cost=dict(type="BBoxBEVL1Cost", weight=1.0),
        iou_cost=dict(type="IoU3DCost", weight=1.0),
        iou_calculator=dict(type="BboxOverlaps3D"),
    ):
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # Hard code here to be compatible with the interface of
        # `ClassificationCost` in mmdet.
        gt_instances, pred_instances = InstanceData(labels=gt_labels), InstanceData(scores=cls_pred[0].T)
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" ' "to install scipy first.")
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        # max_overlaps = iou.max(1).values
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
