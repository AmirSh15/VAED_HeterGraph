import argparse
import builtins
import math
import os
import pickle
import random
import re
import socket
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from backbone.select_backbone import select_backbone
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import utils.augmentation as A
import utils.tensorboard_utils as TB
import utils.transforms as T

plt.switch_backend("agg")

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset.lmdb_dataset import *

from model.pretrain import InfoNCE, UberNCE
from utils.utils import (
    AverageMeter,
    FastDataLoader,
    Logger,
    ProgressMeter,
    batch_denorm,
    calc_mask_accuracy,
    calc_topk_accuracy,
    neq_load_customized,
    save_checkpoint,
    write_log,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="s3d", type=str)  # r18-all
    parser.add_argument("--model", default="infonce", type=str)
    parser.add_argument("--dataset", default="ucf101-2clip", type=str)
    parser.add_argument(
        "--seq_len", default=32, type=int, help="number of frames in each video block"
    )
    parser.add_argument("--num_seq", default=2, type=int, help="number of video blocks")
    parser.add_argument("--ds", default=1, type=int, help="frame down sampling rate")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
    parser.add_argument(
        "--resume", default="", type=str, help="path of model to resume"
    )
    parser.add_argument(
        "--pretrain", default="", type=str, help="path of pretrained model"
    )
    parser.add_argument(
        "--test", default="", type=str, help="path of model to load and pause"
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument(
        "--print_freq",
        default=5,
        type=int,
        help="frequency of printing output during training",
    )
    parser.add_argument("--save_freq", default=1, type=int, help="frequency of eval")
    parser.add_argument(
        "--reset_lr",
        action="store_true",
        help="Reset learning rate when resume training?",
    )
    parser.add_argument("--img_dim", default=128, type=int)
    parser.add_argument("--prefix", default="pretrain", type=str)
    parser.add_argument("--name_prefix", default="", type=str)
    parser.add_argument("-j", "--workers", default=16, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # parallel configs:
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    # for torch.distributed.launch
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )

    # moco specific configs:
    parser.add_argument(
        "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--moco-k",
        default=2048,
        type=int,
        help="queue size; number of negative keys (default: 65536)",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
    )
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.local_rank != -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        assert args.local_rank == -1
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0
    args.gpu = gpu

    if args.distributed:
        if args.local_rank != -1:  # torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
        elif args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    args.print = args.gpu == 0
    # suppress printing if not master
    if args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    ### model ###
    print("=> creating {} model with '{}' backbone".format(args.model, args.net))
    if args.model == "infonce":
        model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    elif args.model == "ubernce":
        model = UberNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    args.num_seq = 2
    print("Re-write num_seq to %d" % args.num_seq)

    args.img_path, args.model_path, args.exp_path = set_path(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ### optimizer ###
    params = []
    for name, param in model.named_parameters():
        params.append({"params": param})

    print("\n===========Check Grad============")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, param.requires_grad)
    print("=================================\n")

    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###
    transform_train = get_transform("train", args)
    train_loader = get_dataloader(
        get_data(transform_train, "train", args), "train", args
    )
    transform_train_cuda = transforms.Compose(
        [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)]
    )
    n_data = len(train_loader.dataset)

    print("===================================")

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
            args.start_epoch = checkpoint["epoch"] + 1
            args.iteration = checkpoint["iteration"]
            best_acc = checkpoint["best_acc"]
            state_dict = checkpoint["state_dict"]

            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                print("[WARNING] resuming training with different weights")
                neq_load_customized(model_without_ddp, state_dict, verbose=True)

            print(
                "=> load resumed checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print("[WARNING] failed to load optimizer state, initialize optimizer")
        else:
            print(
                "[Warning] no checkpoint found at '{}', use random init".format(
                    args.resume
                )
            )

    elif args.pretrain:
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=torch.device("cpu"))
            state_dict = checkpoint["state_dict"]

            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                neq_load_customized(model_without_ddp, state_dict, verbose=True)
            print(
                "=> loaded pretrained checkpoint '{}' (epoch {})".format(
                    args.pretrain, checkpoint["epoch"]
                )
            )
        else:
            print(
                "=> no checkpoint found at '{}', use random init".format(args.pretrain)
            )

    else:
        print("=> train from scratch")

    torch.backends.cudnn.benchmark = True

    # tensorboard plot tools
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, "train"))
    args.train_plotter = TB.PlotterThread(writer_train)

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        _, train_acc = train_one_epoch(
            train_loader, model, criterion, optimizer, transform_train_cuda, epoch, args
        )

        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            # save check_point on rank==0 worker
            if (not args.multiprocessing_distributed and args.rank == 0) or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
            ):
                is_best = train_acc > best_acc
                best_acc = max(train_acc, best_acc)
                state_dict = model_without_ddp.state_dict()
                save_dict = {
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "best_acc": best_acc,
                    "optimizer": optimizer.state_dict(),
                    "iteration": args.iteration,
                }
                save_checkpoint(
                    save_dict,
                    is_best,
                    gap=args.save_freq,
                    filename=os.path.join(args.model_path, "epoch%d.pth.tar" % epoch),
                    keep_all="k400" in args.dataset,
                )

    print("Training from ep %d to ep %d finished" % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(
    data_loader, model, criterion, optimizer, transforms_cuda, epoch, args
):
    batch_time = AverageMeter("Time", ":.2f")
    data_time = AverageMeter("Data", ":.2f")
    losses = AverageMeter("Loss", ":.4f")
    top1_meter = AverageMeter("acc@1", ":.4f")
    top5_meter = AverageMeter("acc@5", ":.4f")
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix="Epoch:[{}]".format(epoch),
    )

    model.train()

    def tr(x):
        B = x.size(0)
        return (
            transforms_cuda(x)
            .view(B, 3, args.num_seq, args.seq_len, args.img_dim, args.img_dim)
            .transpose(1, 2)
            .contiguous()
        )

    tic = time.time()
    end = time.time()

    for idx, (input_seq, label) in tqdm(
        enumerate(data_loader), total=len(data_loader), disable=True
    ):
        data_time.update(time.time() - end)
        B = input_seq.size(0)
        input_seq = tr(input_seq.cuda(non_blocking=True))

        if args.model == "infonce":  # 'target' is the index of self
            output, target = model(input_seq)
            loss = criterion(output, target)
            top1, top5 = calc_topk_accuracy(output, target, (1, 5))

        if args.model == "ubernce":  # 'target' is the binary mask
            label = label.cuda(non_blocking=True)
            output, target = model(input_seq, label)
            # optimize all positive pairs, compute the mean for num_pos and for batch_size
            loss = -(F.log_softmax(output, dim=1) * target).sum(1) / target.sum(1)
            loss = loss.mean()
            top1, top5 = calc_mask_accuracy(output, target, (1, 5))

        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(idx)

        if idx % args.print_freq == 0:
            if args.print:
                args.train_plotter.add_data(
                    "local/loss", losses.local_avg, args.iteration
                )
                args.train_plotter.add_data(
                    "local/top1", top1_meter.local_avg, args.iteration
                )

        args.iteration += 1

    print(
        "({gpu:1d})Epoch: [{0}][{1}/{2}]\t"
        "T-epoch:{t:.2f}\t".format(
            epoch, idx, len(data_loader), gpu=args.rank, t=time.time() - tic
        )
    )

    if args.print:
        args.train_plotter.add_data("global/loss", losses.avg, epoch)
        args.train_plotter.add_data("global/top1", top1_meter.avg, epoch)
        args.train_plotter.add_data("global/top5", top5_meter.avg, epoch)

    return losses.avg, top1_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_transform(mode, args):
    null_transform = transforms.Compose(
        [
            A.RandomSizedCrop(
                size=args.img_dim,
                consistent=False,
                seq_len=args.seq_len,
                bottom_area=0.2,
            ),
            A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
            A.ToTensor(),
        ]
    )

    base_transform = transforms.Compose(
        [
            A.RandomSizedCrop(
                size=args.img_dim,
                consistent=False,
                seq_len=args.seq_len,
                bottom_area=0.2,
            ),
            transforms.RandomApply(
                [
                    A.ColorJitter(
                        0.4,
                        0.4,
                        0.4,
                        0.1,
                        p=1.0,
                        consistent=False,
                        seq_len=args.seq_len,
                    )
                ],
                p=0.8,
            ),
            A.RandomGray(p=0.2, seq_len=args.seq_len),
            transforms.RandomApply(
                [A.GaussianBlur([0.1, 2.0], seq_len=args.seq_len)], p=0.5
            ),
            A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
            A.ToTensor(),
        ]
    )

    # oneclip: temporally take one clip, random augment twice
    # twoclip: temporally take two clips, random augment for each
    # merge oneclip & twoclip transforms with 50%/50% probability
    transform = A.TransformController(
        [
            A.TwoClipTransform(
                base_transform, null_transform, seq_len=args.seq_len, p=0.3
            ),
            A.OneClipTransform(base_transform, null_transform, seq_len=args.seq_len),
        ],
        weights=[0.5, 0.5],
    )
    print(transform)
    return transform


def get_data(transform, mode, args):
    print('Loading data for "%s" mode' % mode)

    if args.dataset == "ucf101-2clip":
        dataset = UCF101LMDB_2CLIP(
            mode=mode,
            transform=transform,
            num_frames=args.seq_len,
            ds=args.ds,
            return_label=True,
        )
    elif args.dataset == "ucf101-f-2clip":
        dataset = UCF101Flow_LMDB_2CLIP(
            mode=mode,
            transform=transform,
            num_frames=args.seq_len,
            ds=args.ds,
            return_label=True,
        )

    elif args.dataset == "k400-2clip":
        dataset = K400_LMDB_2CLIP(
            mode=mode,
            transform=transform,
            num_frames=args.seq_len,
            ds=args.ds,
            return_label=True,
        )
    elif args.dataset == "k400-f-2clip":
        dataset = K400_Flow_LMDB_2CLIP(
            mode=mode,
            transform=transform,
            num_frames=args.seq_len,
            ds=args.ds,
            return_label=True,
        )

    return dataset


def get_dataloader(dataset, mode, args):
    print('Creating data loaders for "%s" mode' % mode)
    train_sampler = data.distributed.DistributedSampler(dataset, shuffle=True)
    if mode == "train":
        data_loader = FastDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        raise NotImplementedError
    print('"%s" dataset has size: %d' % (mode, len(dataset)))
    return data_loader


def set_path(args):
    if args.resume:
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test:
        exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = "log-{args.prefix}/{args.name_prefix}{args.model}_k{args.moco_k}_{args.dataset}-{args.img_dim}_{args.net}_\
bs{args.batch_size}_lr{args.lr}_seq{args.num_seq}_len{args.seq_len}_ds{args.ds}{0}".format(
            "_pt=%s" % args.pretrain.replace("/", "-") if args.pretrain else "",
            args=args,
        )
    img_path = os.path.join(exp_path, "img")
    model_path = os.path.join(exp_path, "model")
    if not os.path.exists(img_path):
        if args.distributed and args.gpu == 0:
            os.makedirs(img_path)
    if not os.path.exists(model_path):
        if args.distributed and args.gpu == 0:
            os.makedirs(model_path)
    return img_path, model_path, exp_path


if __name__ == "__main__":
    """
    Three ways to run (recommend first one for simplicity):
    1. CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
       --nproc_per_node=2 main_nce.py (do not use multiprocessing-distributed) ...

       This mode overwrites WORLD_SIZE, overwrites rank with local_rank
       
    2. CUDA_VISIBLE_DEVICES=0,1 python main_nce.py \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 ...

       Official methods from fb/moco repo
       However, lmdb is NOT supported in this mode, because ENV cannot be pickled in mp.spawn

    3. using SLURM scheduler
    """
    args = parse_args()
    main(args)
