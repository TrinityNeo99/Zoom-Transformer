#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: profile_test.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/6/6 22:48 at PyCharm
"""
import os
import numpy as np
import torch
from torchvision.models import resnet18
import time
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == '__main__':
    model = resnet18(pretrained=False)
    model.eval()
    model.cuda(0)
    dump_input = torch.ones(1, 3, 224, 224).cuda(0)

    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='profile/'),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
    ) as prof:
        for i in range(10):
            print(i)
            outputs = model(dump_input)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
