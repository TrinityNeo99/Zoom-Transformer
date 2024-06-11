#!/bin/bash

#
# Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
# All rights reserved
#

#SBATCH --partition=g078t1

#SBATCH --nodes=1
### 指定该作业需要1个节点数

#SBATCH --ntasks-per-node=8
### 每个节点所运行的进程数为8，最大为64

#SBATCH --time=23:45:00
### 作业的最大运行时间，超过时间后作业资源会被SLURM回收;该时间不能超过分区的最大运行时间

#SBATCH --gres=gpu:2
###（声明需要的GPU数量）【单节点最大申请8个GPU】

#SBATCH --comment keliangchen
### 指定从哪个项目扣费（即导师所在的项目名称，可以在平台上查看，或者咨询导师）

### 程序的执行命令
source ~/.bashrc  ### 初始化环境变量
source  /opt/app/anaconda3/bin/activate pytorch-2.1.0
cd /home/u2023010384/project/action_recognition/Zoom-Transformer
pip install -r ./requirement.txt -i https://mirrors.aliyun.com/pypi/simple/
python main.py --config ./config/ntu/temporal_MoE_Mix_c2.yaml