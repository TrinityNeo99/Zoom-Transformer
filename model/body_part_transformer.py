#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: body_part_transformer.py
@Description: 分割人体左右手关键点
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/5/15 16:09 at PyCharm
"""

import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.angular_feature import Angular_feature
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

from model.zoom_angular_transformer import Temporal_Spatial_Trans_unit, bn_init, TCN_GCN_unit
from model.zoom_angular_transformer import import_class

"""
Graph to represent skeleton joints

Joint sequence same as COCO format: {
    0: nose,
    1: left_eye,
    2: right_eye,
    3: left_ear,
    4: right_ear,
    5: left_shoulder,
    6: right_shoulder,
    7: left_elbow,
    8: right_elbow,
    9: left_wrist,
    10: right_wrist,
    11: left_hip,
    12: right_hip,
    13: left_knee,
    14: right_knee,
    15: left_ankle,
    16: right_ankle
}
"""

arm_index = torch.tensor([5, 6, 7, 8, 9, 10])
other_index = torch.tensor([0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16])


class KeyPointDivide(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("arm_index", arm_index)
        self.register_buffer("other_index", other_index)

    def forward(self, x):
        # x: B C T V
        arm_feature = x[:, :, :, self.arm_index]
        other_feature = x[:, :, :, self.other_index]
        return arm_feature, other_feature


class KeyPointMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("arm_index", arm_index)
        self.register_buffer("other_index", other_index)

    def forward(self, arm_feature, other_feature):
        # x: B C T V
        B, C, T, V1 = arm_feature.shape
        _, _, _, V2 = other_feature.shape
        x = torch.zeros(B, C, T, V1 + V2).cuda()
        x[:, :, :, self.arm_index] = arm_feature
        x[:, :, :, self.other_index] = other_feature
        return x


class BodyPartBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_frames, spatial_heads, temporal_heads, temporal_merge=False):
        super().__init__()
        self.num_frames = num_frames
        if temporal_merge:
            self.merge_blk_in_ch = in_channels * 2
        else:
            self.merge_blk_in_ch = in_channels
        self.arm_block = Temporal_Spatial_Trans_unit(in_channels, out_channels, spatial_heads=spatial_heads,
                                                     temporal_heads=temporal_heads,
                                                     window_size=5, temporal_merge=temporal_merge)
        self.other_block = Temporal_Spatial_Trans_unit(in_channels, out_channels, spatial_heads=spatial_heads,
                                                       temporal_heads=temporal_heads,
                                                       window_size=5, temporal_merge=temporal_merge)
        self.body_block = Temporal_Spatial_Trans_unit(self.merge_blk_in_ch, out_channels, spatial_heads=spatial_heads,
                                                      temporal_heads=temporal_heads,
                                                      window_size=5, temporal_merge=False)  # 前面的两部分已经merge过了
        self.keypoint_divide = KeyPointDivide()
        self.keypoint_merge = KeyPointMerge()
        self.data_bn1 = nn.BatchNorm2d(out_channels)
        bn_init(self.data_bn1, 1)
        self.data_bn2 = nn.BatchNorm2d(out_channels)
        bn_init(self.data_bn2, 1)
        self.data_bn3 = nn.BatchNorm2d(out_channels)
        bn_init(self.data_bn3, 1)

    def forward(self, x):
        # x: B C T V
        arm_feature, other_feature = self.keypoint_divide(x)
        arm_feature = self.arm_block(arm_feature)
        arm_feature = self.data_bn1(arm_feature)
        other_feature = self.other_block(other_feature)
        other_feature = self.data_bn2(other_feature)
        x = self.keypoint_merge(arm_feature, other_feature)
        x = self.body_block(x)
        x = self.data_bn3(x)
        return x


class BodyPartLayer(nn.Module):
    def __init__(self, in_channels, num_frames=100):
        super().__init__()
        self.num_frames = num_frames
        self.l1 = BodyPartBlock(in_channels=in_channels, out_channels=48, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=3)
        self.l2 = BodyPartBlock(in_channels=48, out_channels=48, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=3)
        self.l3 = BodyPartBlock(in_channels=48, out_channels=96, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=3, temporal_merge=True)
        self.l4 = BodyPartBlock(in_channels=96, out_channels=96, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=6)
        self.l5 = BodyPartBlock(in_channels=96, out_channels=192, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=6, temporal_merge=True)
        self.l6 = BodyPartBlock(in_channels=192, out_channels=192, num_frames=self.num_frames, spatial_heads=6,
                                temporal_heads=12)
        pass

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return x


class Model(nn.Module):
    def __init__(self, num_class=15, in_channels=3, num_person=5, num_point=17, num_head=6, graph=None,
                 graph_args=dict(), num_frames=100):
        super().__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.angular_feature = Angular_feature()
        self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        self.tcn_gcn_embedding = TCN_GCN_unit(in_channels, 48, self.graph.A[0],
                                              residual=False)  # only contain A[0] (adjacency matrix)
        self.body_part_layer = BodyPartLayer(in_channels=48)
        self.fc = nn.Linear(192, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        x = self.angular_feature.preprocessing_pingpong_coco(x)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # 这样就达到了共享参数的目的
        x = self.tcn_gcn_embedding(x)
        x = self.body_part_layer(x)  # x: B C T V
        x = x.mean(3).mean(2)  # mean
        x = self.fc(x)
        return x
