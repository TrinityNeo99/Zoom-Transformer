#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved
import os
import wandb
import yaml

from main import sweep_train, sweep_test

sweep_config = {
    'method': 'random'
}
metric = {
    'name': 'Eval Best top-1 acc',
    'goal': 'maximize'
}
sweep_config['metric'] = metric
sweep_config['metric'] = metric
sweep_config['metric'] = metric

sweep_config['parameters'] = {}
# if os.path.exists(default_config_path):
#     with open(default_config_path, 'r') as f:
#         default_arg = yaml.safe_load(f)
#         sweep_config['parameters'].update(default_arg)

# 需要sweep的参数
sweep_config['parameters'].update({
    'base_lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 0.1
    },
    'weight_decay': {
        'distribution': 'uniform',
        'min': 0.0,
        'max': 0.001,
    },
    'batch_size': {
        'distribution': 'q_uniform',
        'q': 8,
        'min': 32,
        'max': 256,
    },
    'optimizer': {
        'values': ['Adam', 'SGD']
    },
    'step': {
        'values': [[30, 40, 50], [20, 30, 40], [30, 40]]
    }
})

# sweep_config['early_terminate'] = {
#     'type':'hyperband',
#     'min_iter':35,
#     'eta':2,
#     's':3
# } #在step=3, 6, 12 时考虑是否剪枝
wandb.login(key="bc22e6220c728740eef0df1af4695d3bd63ec155", force=True)
sweep_id = wandb.sweep(sweep_config, project="sports_action_recognition")
print("sweep initial successfully!", sweep_id)
wandb.agent(sweep_id, sweep_train, count=5)
