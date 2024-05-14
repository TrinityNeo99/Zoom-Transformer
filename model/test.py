#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: test.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/5/13 22:20 at PyCharm
"""
import torch


class modelTest():
    def __init__(self):
        self.window_size = (2, 2)

    def test(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        print(relative_position_index)
        print(relative_position_index.shape)
        print(relative_position_index.view(-1))

    def mytest(self):
        coords = torch.arange(4)
        relative_distance = torch.transpose(torch.arange(4).unsqueeze(0), 0, 1)
        reversed_coords = torch.flip(coords, [0])
        _, relative_index = torch.meshgrid([reversed_coords, reversed_coords])
        relative_index = relative_index + relative_distance
        print(relative_index)


if __name__ == '__main__':
    m = modelTest()
    m.test()
    # m.mytest()
