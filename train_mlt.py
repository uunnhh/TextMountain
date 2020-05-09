import os
import time
import random
import argparse
from distutils.version import LooseVersion

import torch
import torch.nn as nn
from models import ModelBuilder, SegmentationModule,Module_textmountain
from utils import AverageMeter,AverageMeter_dict
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
import pdb
from dataset_train_mlt import user_scattered_collate,TrainDataset

def train_epoch(segmentation_module, iterator, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()

    ave_loss_dict=AverageMeter_dict()
    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    for i in range(args.epoch_iters):
        batch_data_0 = next(iterator)
        batch_data_0=batch_data_0[0]
        batch_data=batch_data_0
        torch.cuda.synchronize()
        
        
        feed_dict=batch_data
        feed_dict['images']=feed_dict['images'].cuda()
        feed_dict['score_maps']=feed_dict['score_maps'].cuda()
        feed_dict['training_masks']=feed_dict['training_masks'].cuda()
        torch.cuda.synchronize()
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()
        loss,loss_dict= segmentation_module(batch_data)
        loss=loss.mean()
        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()


        # update average loss and acc
        ave_total_loss.update(loss.data.item())

        for key in loss_dict:
            loss_dict[key]=loss_dict[key].data.item()
        ave_loss_dict.update(loss_dict)
        batch_time.update(time.time() - tic)
        tic = time.time()
        # calculate accuracy, and display
        
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  ' Loss: {:.6f},  '
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, args.running_lr_decoder,
                          ave_total_loss.average()))

            loss_print='  '
            loss_dict=ave_loss_dict.average()
            for key in loss_dict:
                loss_print=loss_print+'{}:  {:.6f}  '.format(key,loss_dict[key])
            print(loss_print)

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters

        cur_iter = i + (epoch - 1) * args.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, args)



def checkpoint(optimizers,nets, history, args, epoch_num):
    print('Saving checkpoints...')
    data_set='MSRA'
    (net_encoder, net_decoder) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()


    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_latest))
    torch.save(dict_decoder,
               '{}/decoder_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (net_encoder, net_decoder) = nets
    
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)

    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = ((1. - float(cur_iter) / args.max_iters) ** args.lr_pow)
    args.running_lr_encoder = args.lr_encoder * scale_running_lr
    args.running_lr_decoder = args.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = args.running_lr_decoder


def main(args):
    # Network Builders
    builder = ModelBuilder()
    encoder_fn=builder.build_encoder_textmountain
    decoder_fn=builder.build_decoder_textmountain

    dataset_train = TrainDataset(batch_per_gpu=12
       )

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=1,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=int(2),
        drop_last=True,
        pin_memory=False)
    iterator_train = iter(loader_train)


    net_encoder = encoder_fn(
        )
    net_decoder = decoder_fn()

    segmentation_module=Module_textmountain(
    	net_encoder,net_decoder)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder)
    optimizers = create_optimizers(nets, args)

    # Main loop

    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train_epoch(segmentation_module, iterator_train, optimizers, history, epoch, args)

        # checkpointing
    checkpoint(optimizers,nets, history, args, epoch)
    
    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")



    # optimization related arguments
    parser.add_argument('--num_gpus', default=1, type=int,
                        help='number of gpus to use, must be set to 1')

    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=9000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=5e-3, type=float, help='LR')#2e-3
    parser.add_argument('--lr_decoder', default=5e-3, type=float, help='LR')#
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')

    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')




    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt/',
                        help='folder to output checkpoints')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')


    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.max_iters = args.epoch_iters * args.num_epoch
    args.running_lr_encoder = args.lr_encoder
    args.running_lr_decoder = args.lr_decoder


    args.id += '-ngpus' + str(args.num_gpus)




    args.id += '-LR_encoder' + str(args.lr_encoder)
    args.id += '-LR_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    args.id += '-fixBN' + str(args.fix_bn)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
