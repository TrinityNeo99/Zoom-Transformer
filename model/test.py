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

    def test_discrete(self):
        # 创建一个5x3的随机张量，元素在0到1之间
        input_tensor = torch.rand(5, 5)

        # 指定离散化的bins
        bins = torch.tensor([0.3, 0.6, 0.9])

        # 调用函数进行离散化
        discretized_tensor = discretize(input_tensor, bins)

        print("原始张量:\n", input_tensor)
        print("离散化后的张量:\n", discretized_tensor)


def discretize(tensor, bins):
    # 创建一个张量来存储离散化的结果
    discretized = torch.zeros_like(tensor)

    # 遍历每一个可能的bins值，计算与bins中每个值的差的绝对值
    for b in bins:
        # 计算当前bin与tensor中所有元素的绝对差值
        abs_diff = torch.abs(tensor - b)
        # 如果当前的bin比之前的更接近，则更新离散化结果
        discretized[abs_diff < torch.abs(discretized - tensor)] = b

    return discretized


if __name__ == '__main__':
    m = modelTest()
    # m.test()
    # m.mytest()
    m.test_discrete()
