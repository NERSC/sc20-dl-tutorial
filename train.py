import os
import time
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

import models.resnet
from utils.YParams import YParams
from utils.cifar100_data_loader import get_data_loader

class Trainer():

  def __init__(self, params):
    self.params = params
    self.device = torch.cuda.current_device()

    # first constrcut the dataloader on rank0 in case the data is not downloaded
    if params.world_rank == 0:
      logging.info('rank %d, begin data loader init'%params.world_rank)
      self.train_data_loader, self.train_sampler = get_data_loader(params, params.data_path, dist.is_initialized(), is_train=True)
      self.valid_data_loader, self.valid_sampler = get_data_loader(params, params.data_path, dist.is_initialized(), is_train=False)
      logging.info('rank %d, data loader initialized'%params.world_rank)

    # wait for rank0 to finish downloading the data
    if dist.is_initialized():
      dist.barrier()

    # now construct the dataloaders on other ranks
    if params.world_rank != 0:
      logging.info('rank %d, begin data loader init'%params.world_rank)
      self.train_data_loader, self.train_sampler = get_data_loader(params, params.data_path, dist.is_initialized(), is_train=True)
      self.valid_data_loader, self.valid_sampler = get_data_loader(params, params.data_path, dist.is_initialized(), is_train=False)
      logging.info('rank %d, data loader initialized'%params.world_rank)

    self.model = models.resnet.resnet50(num_classes=params.num_classes).to(self.device)

    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=params.lr,
                                     momentum=params.momentum, weight_decay=params.weight_decay)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=10, mode='min')
    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank],
                                           output_device=[params.local_rank])
    self.iters = 0
    self.startEpoch = 0
    if params.resuming:
      logging.info("Loading checkpoint %s"%params.checkpoint_path)
      self.restore_checkpoint(params.checkpoint_path)
    self.epoch = self.startEpoch

    if params.log_to_screen:
      logging.info(self.model)

    if params.log_to_tensorboard:
      self.writer = SummaryWriter(os.path.join(params.experiment_dir, 'tb_logs'))

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
        self.valid_sampler.set_epoch(epoch)

      if epoch < params.lr_warmup_epochs:
        self.optimizer.param_groups[0]['lr'] = params.lr*float(epoch+1.)/float(params.lr_warmup_epochs)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()
      if epoch >= params.lr_warmup_epochs:
        self.scheduler.step(valid_logs['loss'])

      if self.params.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path)

      if self.params.log_to_tensorboard:
        self.writer.add_scalar('loss/train', train_logs['loss'], self.epoch) 
        self.writer.add_scalar('loss/valid', valid_logs['loss'], self.epoch) 
        self.writer.add_scalar('acc1/train', train_logs['acc1'], self.epoch) 
        self.writer.add_scalar('acc1/valid', valid_logs['acc1'], self.epoch) 
        self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.epoch)

      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info('train data time={}, train time={}, valid step time={}, train acc1={}, valid acc1={}'.format(data_time, tr_time,
                                                                                                                  valid_time,
                                                                                                                  train_logs['acc1'],
                                                                                                                  valid_logs['acc1']))

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    for i, data in enumerate(self.train_data_loader, 0):
      self.iters += 1
      data_start = time.time()
      images, labels = map(lambda x: x.to(self.device), data)
      data_time += time.time() - data_start

      tr_start = time.time()
      self.model.zero_grad()
      self.model.train()
      outputs = self.model(images)
      loss = self.criterion(outputs, labels)
      loss.backward()
      self.optimizer.step()
      tr_time += time.time() - tr_start

    # save metrics of last batch
    _, preds = outputs.max(1)
    acc1 = preds.eq(labels).sum().float()/labels.shape[0]
    logs = {'loss': loss,
            'acc1': acc1}

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()

    valid_start = time.time()
    loss = 0.0
    correct = 0.0
    with torch.no_grad():
      for data in self.valid_data_loader:
        images, labels = map(lambda x: x.to(self.device), data)
        outputs = self.model(images)
        loss += self.criterion(outputs, labels)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().float()/labels.shape[0]

    logs = {'loss': loss/len(self.valid_data_loader),
            'acc1': correct/len(self.valid_data_loader)}
    valid_time = time.time() - valid_start

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        logs[key] = torch.as_tensor(logs[key]).to(self.device)
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    return valid_time, logs

  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
    self.model.load_state_dict(checkpoint['model_state'])
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch'] + 1
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--local_rank", default=0, type=int)
  parser.add_argument("--yaml_config", default='./config/cifar100.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)

  # setup distributed training variables and intialize cluster if using
  params['world_size'] = 1
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])

  params['local_rank'] = args.local_rank
  params['world_rank'] = 0
  if params['world_size'] > 1:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    params['world_rank'] = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  torch.backends.cudnn.benchmark = True

  # setup output directory
  expDir = os.path.join('./expts', args.config)
  if params.world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir)
      os.makedirs(os.path.join(expDir, 'checkpoints/'))

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'checkpoints/ckpt.tar')
  params['resuming'] = True if os.path.isfile(params.checkpoint_path) else False

  if params.world_rank==0:
    params.log()
  params['log_to_screen'] = params.log_to_screen and params.world_rank==0
  params['log_to_tensorboard'] = params.log_to_tensorboard and params.world_rank==0

  trainer = Trainer(params)
  trainer.train()
