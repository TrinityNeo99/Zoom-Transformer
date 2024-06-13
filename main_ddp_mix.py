#!/usr/bin/env python

#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import wandb
import math
from datetime import datetime
# from torchstat import stat
from thop import profile as tprofile
from thop import clever_format

sys.path.append("../")
from Evaluate.evaluate import generate_confusion_matrix

# from torch.profiler import profile, record_function, ProfilerActivity

# torch.set_num_threads(2)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ddp_function import distributed_concat, SequentialDistributedSampler

dist.init_process_group(backend='nccl')
from torch.cuda.amp import GradScaler, autocast


# from thop import clever_format
# from thop import profile

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(seed=0):
    fix_seed = seed + 1
    torch.cuda.manual_seed_all(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    random.seed(fix_seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='TT')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=1,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--gamma', type=float, default=0.1, help='scaler of learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--dataset', default="p2a-14")

    # DDP support
    parser.add_argument("--DDP", default=False, action='store_true')
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    # answer = input('delete it? y/n:')
                    answer = 'y'
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        # input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                # self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                # self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                # self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
                pass

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc5 = 0
        self.scaler = GradScaler()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train' and self.arg.DDP:
            train_dataset = Feeder(**self.arg.train_feeder_args)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
                sampler=self.train_sampler,
                pin_memory=True)

        test_dataset = Feeder(**self.arg.test_feeder_args)
        self.test_sampler = SequentialDistributedSampler(test_dataset, batch_size=self.arg.test_batch_size)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
            sampler=self.test_sampler,
            pin_memory=True)

    def load_model(self):
        local_rank = self.arg.local_rank
        torch.cuda.set_device(self.arg.local_rank)
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).to(local_rank)
        self.loss = nn.CrossEntropyLoss().to(local_rank)

        if self.arg.weights:
            current_epoch = int(os.path.basename(self.arg.weights).split("-")[-1].replace(".pt", ""))
            self.arg.start_epoch = current_epoch
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.to(local_rank)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        self.calculate_params_flops(3, self.arg.model_args['num_frame'], self.arg.model_args['num_point'],
                                    self.arg.model_args['num_person'], Model(**self.arg.model_args).to(local_rank))
        # self.profile_model(3, self.arg.model_args['num_frame'], self.arg.model_args['num_point'],
        #                    self.arg.model_args['num_person'], Model(**self.arg.model_args).cuda(output_device))

        if self.arg.DDP:
            self.model = self.model.to(local_rank)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank,
                             find_unused_parameters=False)
        else:
            raise Exception("not DDP")

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch, gamma):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=True):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch, self.arg.gamma)
        loss_value = []
        accs = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
        #                                      profile_memory=False) as prof:
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='profile/'),
        #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True,
        # ) as prof:
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            data = data.float().to(self.arg.local_rank)
            label = label.long().to(self.arg.local_rank)
            timer['dataloader'] += self.split_time()
            with autocast():
                # forward
                output = self.model(data)
                if isinstance(output, tuple):
                    output, aux_loss = output
                    aux_loss = aux_loss.mean()
                else:
                    aux_loss = 0
                loss = self.loss(output, label) + aux_loss

            # backward

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            grad_clip = False
            if grad_clip:
                max_norm = 1.0
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            accs.append(acc)
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            timer['statistics'] += self.split_time()
            # prof.step()
            if dist.get_rank() == 0:
                wandb.log({"train_step_loss": loss.data.item()})
            # profile
            # if batch_idx > 10:
            #     break
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=10))
        # exit(0)
        accs = torch.tensor(accs).data.cpu().numpy()
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))
        if dist.get_rank() == 0:
            wandb.log({"train_total_loss": np.mean(loss_value), "epoch": epoch})
            wandb.log({f"train_acc_top_1": 100 * np.mean(accs), "epoch": epoch})
            wandb.log({"runing_lr": self.lr, "epoch": epoch})

        if epoch % self.arg.save_interval == 0 and save_model and dist.get_rank() == 0:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, os.path.join(self.arg.work_dir, self.arg.model_saved_name + '-' + str(epoch) + '.pt'))

    def eval(self, epoch, save_score=False, loader_name=['test']):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            labels = []
            predictions = []
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    with autocast():
                        data = data.float().to(self.arg.local_rank)
                        label = label.long().to(self.arg.local_rank)
                        output = self.model(data)
                        if isinstance(output, tuple):
                            output, aux_loss = output
                            aux_loss = aux_loss.mean()
                        else:
                            aux_loss = 0
                        loss = self.loss(output, label)
                        predictions.append(output)
                        labels.append(label)
                        loss_value.append(loss.data.item())
                    # if batch_idx > 10:
                    #     break

            predictions = distributed_concat(torch.cat(predictions, dim=0),
                                             len(self.test_sampler.dataset))
            labels = distributed_concat(torch.cat(labels, dim=0),
                                        len(self.test_sampler.dataset))
            scores = predictions.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            accuracy = self.data_loader[ln].dataset.top_k(scores, 1, labels)
            accuracy_top5 = self.data_loader[ln].dataset.top_k(scores, 5, labels)
            if self.arg.dataset is not None:
                dataset = self.arg.dataset
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                predicted_labels = scores.argsort()[:, -1]
                if dist.get_rank() == 0:
                    generate_confusion_matrix(predicted_labels, labels, dataset=dataset,
                                              output_dir=self.arg.work_dir, epoch=epoch)
            if accuracy_top5 > self.best_acc5:
                self.best_acc5 = accuracy_top5

            mean_loss = np.mean(loss_value)
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', round(accuracy * 100, 2), ' model: ', self.arg.model_saved_name)
            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name[: len(scores)], scores))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), mean_loss))

            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(scores, k, labels)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            if dist.get_rank() == 0:
                wandb.log({"eval_total_loss": mean_loss, "epoch": epoch})
                wandb.log({"Eval Best top-1 acc": 100 * self.best_acc, "epoch": epoch})
                wandb.log({"Eval Best top-5 acc": 100 * self.best_acc5, "epoch": epoch})
                for k in self.arg.show_topk:
                    wandb.log(
                        {f"eval_acc_top_{k}": 100 * self.data_loader[ln].dataset.top_k(scores, 1, labels),
                         "epoch": epoch})

    def calculate_params_flops(self, in_channel, num_frame, num_keypoint, num_person, model):
        # N, C, T, V, M
        dummy_input = torch.randn(1, in_channel, num_frame, num_keypoint, num_person).to(self.arg.local_rank)
        flops, params = tprofile(model, inputs=(dummy_input,))
        # flops, params = clever_format([flops, params], '%.3f')
        flops = round(flops / (10 ** 9), 2)
        params = round(params / (10 ** 6), 2)
        if dist.get_rank() == 0:
            wandb.log({"model_params": params})
            wandb.log({"model_flops": flops})
        del model
        del dummy_input

    def profile_model(self, in_channel, num_frame, num_keypoint, num_person, model):
        print(self.output_device)
        dummy_input = torch.randn(2, in_channel, num_frame, num_keypoint, num_person).to(self.arg.local_rank)
        # Warn-up
        for _ in range(50):
            start = time.time()
            outputs = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            print('Time:{}ms'.format((end - start) * 1000))
        with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True,
                                             profile_memory=False) as prof:
            outputs = model(dummy_input)
        print(prof.table())
        prof.export_chrome_trace('./profile.json')
        del model
        del dummy_input

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train_sampler.set_epoch(epoch)  # 利用 epoch shuffle 数据
                self.train(epoch, save_model=True)
                if epoch % self.arg.eval_interval == 0:
                    self.eval(
                        epoch,
                        save_score=self.arg.save_score,
                        loader_name=['test'])
            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def wandb_init(args):
    wandb.login(key="610ea58ece04cbfb08fe53c2d852fccf1833d910", force=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="action_recognition",
        # name="ASE_GCN_baseline",
        name=args.model_saved_name,
        # track hyperparameters and run metadata
        config=args
    )


def sweep_test():
    wandb.init()
    bacc = 0
    for i in range(50):
        acc = random.randint(0, 100)
        if acc > bacc:
            bacc = acc
    wandb.log({"Eval Best top-1 acc": bacc})


def sweep_train():
    default_config_path = "./config/p2a-v1/train_2024-5-5_data_angular.yaml"
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    p.config = default_config_path
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    if not os.path.exists(arg.work_dir):
        os.mkdir(arg.work_dir)
    arg.timestamp = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    current_work_dir = os.path.join(arg.work_dir, arg.model_saved_name, arg.timestamp)
    os.makedirs(current_work_dir, exist_ok=True)
    init_seed(0)
    arg.work_dir = current_work_dir
    if dist.get_rank() == 0:
        wandb_init(args=arg)
    processor = Processor(wandb.config)
    processor.start()
    wandb.finish()


def bodypart_sweep_train():
    default_config_path = "./config/p2a-v1/train_bodypart.yaml"
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    p.config = default_config_path
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    if not os.path.exists(arg.work_dir):
        os.mkdir(arg.work_dir)
    arg.timestamp = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    current_work_dir = os.path.join(arg.work_dir, arg.model_saved_name, arg.timestamp)
    os.makedirs(current_work_dir)
    init_seed(0)
    arg.work_dir = current_work_dir
    wandb_init(args=arg)
    processor = Processor(wandb.config)
    processor.start()
    wandb.finish()


# wandb.mode = "offline"

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    if not os.path.exists(arg.work_dir):
        os.makedirs(arg.work_dir, exist_ok=True)
    arg.timestamp = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
    current_work_dir = os.path.join(arg.work_dir, arg.model_saved_name, arg.timestamp)
    os.makedirs(current_work_dir, exist_ok=True)
    init_seed(0)
    arg.work_dir = current_work_dir
    if dist.get_rank() == 0:
        wandb_init(args=arg)
    print("num_workers: ", arg.num_worker)
    processor = Processor(arg)
    processor.start()
    wandb.finish()
