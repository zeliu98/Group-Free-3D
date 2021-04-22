import torch
import torch.nn as nn
import numpy as np
import sys
import os

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
import pointnet2_utils


class PointsObjClsModule(nn.Module):
    def __init__(self, seed_feature_dim):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv3 = torch.nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class FPSModule(nn.Module):
    def __init__(self, num_proposal):
        super().__init__()
        self.num_proposal = num_proposal

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        # Farthest point sampling (FPS)
        sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.num_proposal)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

        return new_xyz, new_features, sample_inds


class PredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_class_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster, 1)
        self.size_residual_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster * 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        mean_size_arr = torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster, 3)
        mean_size_arr = mean_size_arr.unsqueeze(0).unsqueeze(0)  # (1, 1, num_size_cluster, 3)
        size_scores = self.size_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_size_cluster)
        size_residuals_normalized = self.size_residual_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, self.num_size_cluster, 3])  # (batch_size, num_proposal, num_size_cluster, 3)
        size_residuals = size_residuals_normalized * mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        size_recover = size_residuals + mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        pred_size_class = torch.argmax(size_scores, -1)  # batch_size, num_proposal
        pred_size_class = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_size = torch.gather(size_recover, 2, pred_size_class)  # batch_size, num_proposal, 1, 3
        pred_size = pred_size.squeeze_(2)  # batch_size, num_proposal, 3

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}size_scores'] = size_scores
        end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
        end_points[f'{prefix}size_residuals'] = size_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        # # used to check bbox size
        # l = pred_size[:, :, 0]
        # h = pred_size[:, :, 1]
        # w = pred_size[:, :, 2]
        # x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], -1)  # N Pq 8
        # y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], -1)  # N Pq 8
        # z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1)  # N Pq 8
        # corners = torch.stack([x_corners, y_corners, z_corners], -1)  # N Pq 8 3
        # bbox = center.unsqueeze(2) + corners
        # end_points[f'{prefix}bbox_check'] = bbox
        return center, pred_size


class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_pred_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return center, pred_size
