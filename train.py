import os
import time
import random
import importlib
import shutil
import warnings
import math

import torch
import torch.nn as nn
import numpy as np

from utils.utils import Logger, save_model, smallest_greater, NaNLossError
from utils.getters import get_meters, get_lr_scheduler, get_optimizer
from utils.model_profiling import profiling
from utils.plots import plot_acc_width
from utils.meters import ScalarMeter, flush_wm_meters
from utils.config import load_yml_config
from utils.data_prep import prepare_data

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================================== #
# ============================= Print functions ============================= #

def print_loss_acc(results, batch_idx, num_batches):
    msg = '{:.1f}%'.format(100. * (batch_idx + 1) / num_batches)
    for wm in sorted(results.keys(), reverse=True):
        msg += ' [{:.4f}, {:.4f}, {:.2f}%]'.format(
            float(wm), results[wm]['loss'], results[wm]['acc1'] * 100)
    print(msg)
    return


def print_meters(meters, phase, total):
    sum_error1 = 0
    results = flush_wm_meters(meters, flush=True)
    print()
    for wm in sorted(results.keys(), reverse=True):
        print('==> {} width {:.4f},'.format(phase, float(wm)),
              'Loss: {:.6f},'.format(results[wm]['loss']),
              'Accuracy: {}/{} ({:.2f}%),'.format(
                   int(total * results[wm]['acc1']),
                   total, results[wm]['acc1'] * 100),
              'Top5 Acc: {}/{} ({:.2f}%),'.format(
                   int(total * results[wm]['acc5']),
                   total, results[wm]['acc5'] * 100))
        sum_error1 += results[wm]['err1']
    return sum_error1 / len(results), results


# =========================================================================== #
# ============================= Main functions ============================== #

def forward_loss(model, criterion, input, target, meter, topk=[1, 5],
                 verbose=False):
    """ Perform one forward pass """

    # Forward
    output = model(input)
    loss = torch.mean(criterion(output, target))
    meter['loss'].cache(loss.item())

    # Record results
    _, preds = output.topk(max(topk))
    pred = preds.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in topk:
        correct_k = correct[:k].float().sum(0)
        error_list = list(1. - correct_k.cpu().detach().numpy())
        meter['top{}_error'.format(k)].cache_list(error_list)

    if math.isnan(loss):
        raise NaNLossError

    if verbose:
        return loss, output, preds
    return loss


def rand_width_mults(width_mult_args):
    """ Generate randomly sampled width factors """

    mode = getattr(width_mult_args, 'mode', 'N/A')
    intervals = sorted(getattr(width_mult_args, 'intervals', [0., 1.]))
    min_w = min(intervals)
    max_w = max(intervals)

    # Sample width values using the specified strategy
    if mode == 'n-random':
        train_widths = np.random.uniform(min_w, max_w, width_mult_args.num)
    elif mode == '1-random-per-interval':
        train_widths = [random.uniform(intervals[i - 1], s)
                        for i, s in enumerate(intervals) if i > 0]
    elif mode == 'sandwich':
        # US-Net's scheme. See https://arxiv.org/abs/1903.05134
        train_widths = ([min_w, max_w] + list(np.random.uniform(
            min_w, max_w, width_mult_args.num - 2)))
    else:
        raise NotImplementedError('Sampling mode', mode, 'not found!')
    return train_widths


def test(epoch, loader, model, criterion, meters, width_mults, topk=[1, 5]):
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            target = target.cuda(non_blocking=True)
            for width_mult in sorted(width_mults, reverse=True):
                meter = meters[str(width_mult)]
                model.module.set_width_mult(width_mult) # Set active width
                forward_loss(model, criterion, input, target, meter, topk)

    avg_error1, results = print_meters(meters, 'Test', len(loader.dataset))
    return avg_error1, results


def train(epoch, num_epochs, loader, model, criterion, optimizer, meters,
          width_mults, log_interval, topk,
          rand_width_mult_args=None, lr_decay_per_step=None):
    model.train()
    for batch_idx, (input, target) in enumerate(loader):
        if lr_decay_per_step is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] -= lr_decay_per_step

        # Train
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()

        meter_width_mults = width_mults
        train_width_mults = width_mults
        if rand_width_mult_args: # Random-sample training
            train_width_mults = rand_width_mults(rand_width_mult_args)

        for width_mult in sorted(train_width_mults, reverse=True):
            key = smallest_greater(width_mult, meter_width_mults)
            meter = meters[str(key)]

            model.module.set_width_mult(width_mult) # Set active width
            loss, output, preds = forward_loss(
                model, criterion, input, target, meter, topk, verbose=True)
            loss.backward()

        optimizer.step()

        # Print status
        if (batch_idx % log_interval == 0 or batch_idx == len(loader) - 1):
            results = flush_wm_meters(meters, flush=False)
            print_loss_acc(results, batch_idx, num_batches=len(loader))

    _, results = print_meters(meters, 'Train', len(loader.dataset))
    return results


def run_one_experiment():
    t_exp_start = time.time()

    # Save all print-out to a logger file
    logger = Logger(FLAGS.log_file)

    # Print experience setup
    for k in sorted(FLAGS.keys()):
        print('{}: {}'.format(k, FLAGS[k]))

    # Init torch
    if FLAGS.seed is None:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Init model
    model = importlib.import_module(FLAGS.module_name).get_model(FLAGS)
    model = torch.nn.DataParallel(model).cuda()

    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        model.module.load_state_dict(checkpoint['model'])
        print('Loaded model {}.'.format(FLAGS.pretrained))

    if FLAGS.model_profiling and len(FLAGS.model_profiling) > 0:
        print(model)
        profiling(model, FLAGS.model_profiling, FLAGS.image_size,
                  FLAGS.image_channels, FLAGS.train_width_mults,
                  FLAGS.model_profiling_verbose)
    logger.flush()

    # Init data loaders
    train_loader, val_loader, _, train_set = prepare_data(
        FLAGS.dataset, FLAGS.data_dir, FLAGS.data_transforms,
        FLAGS.data_loader, FLAGS.data_loader_workers, FLAGS.train_batch_size,
        FLAGS.val_batch_size, FLAGS.drop_last, FLAGS.test_only)
    class_labels = train_set.classes

    # Perform inference/test only
    if FLAGS.test_only:
        print('Start testing...')
        min_wm = min(FLAGS.train_width_mults)
        max_wm = max(FLAGS.train_width_mults)
        if FLAGS.test_num_width_mults == 1:
            test_width_mults = []
        else:
            step = (max_wm - min_wm) / (FLAGS.test_num_width_mults - 1)
            test_width_mults = np.arange(min_wm, max_wm, step).tolist()
        test_width_mults += [max_wm]

        criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        test_meters = get_meters('val', FLAGS.topk, test_width_mults)
        epoch = -1

        avg_error1, _ = test(epoch, val_loader, model, criterion, test_meters,
                             test_width_mults, topk=FLAGS.topk)
        print('==> Epoch avg accuracy {:.2f}%,'.format((1 - avg_error1) * 100))

        logger.close()
        plot_acc_width(FLAGS.log_file)
        return

    # Init training devices
    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    optimizer = get_optimizer(model, FLAGS.optimizer, FLAGS.weight_decay,
                              FLAGS.lr, FLAGS.momentum, FLAGS.nesterov,
                              depthwise=FLAGS.depthwise)
    lr_scheduler = get_lr_scheduler(optimizer, FLAGS.lr_scheduler,
                                    FLAGS.lr_scheduler_params)

    train_meters = get_meters('train', FLAGS.topk, FLAGS.train_width_mults)
    val_meters = get_meters('val', FLAGS.topk, FLAGS.train_width_mults)
    val_meters['best_val_error1'] = ScalarMeter('best_val_error1')

    time_meter = ScalarMeter('runtime')

    # Perform training
    print('Start training...')
    last_epoch = -1
    best_val_error1 = 1.
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        t_epoch_start = time.time()
        print('\nEpoch {}/{}.'.format(epoch + 1, FLAGS.num_epochs)
              + ' Print format: [width factor, loss, accuracy].'
              + ' Learning rate: {}'.format(optimizer.param_groups[0]['lr']))

        # Train one epoch
        steps_per_epoch = len(train_loader.dataset) / FLAGS.train_batch_size
        total_steps = FLAGS.num_epochs * steps_per_epoch
        lr_decay_per_step = (None if FLAGS.lr_scheduler != 'linear_decaying'
                             else FLAGS.lr / total_steps)
        if FLAGS.lr_scheduler == 'linear_decaying':
            lr_decay_per_step = (
                FLAGS.lr / FLAGS.num_epochs /
                len(train_loader.dataset) * FLAGS.train_batch_size)
        train_results = train(
            epoch, FLAGS.num_epochs, train_loader, model, criterion,
            optimizer, train_meters, FLAGS.train_width_mults,
            FLAGS.log_interval, FLAGS.topk,
            FLAGS.rand_width_mult_args, lr_decay_per_step)

        # Validate
        avg_error1, val_results = test(
            epoch, val_loader, model, criterion, val_meters,
            FLAGS.train_width_mults, topk=FLAGS.topk)

        # Update best result
        is_best = avg_error1 < best_val_error1
        if is_best:
            best_val_error1 = avg_error1
        val_meters['best_val_error1'].cache(best_val_error1)

        # Save checkpoint
        print()
        if FLAGS.saving_checkpoint:
            save_model(model, optimizer, epoch, FLAGS.train_width_mults,
                       FLAGS.rand_width_mult_args, train_meters, val_meters,
                       1 - avg_error1, 1 - best_val_error1,
                       FLAGS.epoch_checkpoint, is_best, FLAGS.best_checkpoint)
        print('==> Epoch avg accuracy {:.2f}%,'.format((1 - avg_error1) * 100),
              'Best accuracy: {:.2f}%\n'.format((1 - best_val_error1) * 100))

        logger.flush()

        if lr_scheduler is not None and epoch != FLAGS.num_epochs - 1:
            lr_scheduler.step()
        print('Epoch time: {:.4f} mins'.format(
            (time.time() - t_epoch_start) / 60))

    print('Total time: {:.4f} mins'.format((time.time() - t_exp_start) / 60))
    logger.close()
    return


def preprocess_args(args, cfg_file):
    """ Set default parameters and formating """

    print('Experiment name:', args.experiment)

    args.test_only = getattr(args, 'test_only', False)

    # Logger
    args.log_dir = os.path.join(args.log_dir, args.experiment) + '/'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    phase = '_test' if args.test_only else '_train'
    args.log_file = args.log_dir + args.experiment + phase + '_log.txt'
    args.best_checkpoint = args.log_dir + args.experiment + '_best_model.t7'
    args.epoch_checkpoint = args.log_dir + args.experiment + '_epoch_ckpt.t7'
    args.batch_checkpoint = args.log_dir + args.experiment + '_batch_ckpt.t7'

    if not args.test_only:
        shutil.copyfile(cfg_file, args.log_dir + os.path.basename(cfg_file))

    # Model
    args.model_profiling_verbose = getattr(
        args, 'model_profiling_verbose', True)

    args.pretrained = getattr(args, 'pretrained', '')
    if args.test_only and not args.pretrained:
        args.pretrained = args.best_checkpoint
    args.test_num_width_mults = getattr(args, 'test_num_width_mults', 10)

    # Data
    args.drop_last = getattr(args, 'drop_last', False)

    # Training
    args.seed = getattr(args, 'seed', None)
    args.rand_width_mult_args = getattr(args, 'rand_width_mult_args', None)
    args.tracking_bn_stats = getattr(args, 'tracking_bn_stats', True)
    args.saving_checkpoint = getattr(args, 'saving_checkpoint', True)
    args.use_tensorboard = getattr(args, 'use_tensorboard', False)

    if args.lr_scheduler in ['exp_decaying', 'linear_decaying']:
        args.lr_scheduler_params.num_epochs = args.num_epochs
    elif args.lr_scheduler == 'multistep':
        milestones = args.lr_scheduler_params.multistep_lr_milestones
        if milestones and isinstance(milestones[0], float):
            args.lr_scheduler_params.multistep_lr_milestones = (
                [int(ms * args.num_epochs) for ms in milestones])

    if args.resume:
        raise NotImplementedError
    return args


def main():
    global FLAGS
    args, cfg_file = load_yml_config()
    FLAGS = preprocess_args(args, cfg_file)
    run_one_experiment()
    return


if __name__ == "__main__":
    main()

