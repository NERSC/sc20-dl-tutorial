import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data_loader(params, files_pattern, distributed, is_train):

  if is_train:
    transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(params.rnd_rotation_angle),
      transforms.ToTensor(),
      transforms.Normalize(tuple(params.cifar100_mean),
                           tuple(params.cifar100_std))])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(tuple(params.cifar100_mean),
                           tuple(params.cifar100_std))])

  dataset = datasets.CIFAR100(root=params.data_path,
                              train=is_train,
                              download=True if (is_train and params.world_rank==0) else False,
                              transform=transform)

  sampler = DistributedSampler(dataset, shuffle=True) if distributed else None

  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size) if is_train else int(params.valid_batch_size_per_gpu),
                          num_workers=params.num_data_workers,
                          shuffle=(sampler is None),
                          sampler=sampler,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  return dataloader, sampler
