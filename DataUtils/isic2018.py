import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torch

import torchvision.datasets as dset
import torchvision.transforms as T

# 图像大小：600*450
# 个数：10015
# 种类：7(0~6)

DATAPATH = os.environ['datapath']
NUM_VAL = 1000
NUM_TRAIN = 10015 - NUM_VAL

def getlabels(mode):
  if mode=='Train':
    df = pd.read_csv(DATAPATH + 'trainlabel.csv')
  else:
    df = pd.read_csv(DATAPATH + 'testlabel.csv')
  images = np.array(df['image'].values.tolist())
  labels = np.array(list(range(0, len(images))))
  label_dic = df.columns.values.tolist()[1:]
  val_ind = np.array([])

  for i in range(0, len(label_dic)):
    ind = df.index[df[df.columns[i+1]] == 1].tolist()
    lab = label_dic[i]
    labels[ind] = i
    if i != len(label_dic)-1:
      val_ind = np.append(np.random.choice(ind, int(NUM_VAL*float(len(ind)/len(labels)))), val_ind)
    else:
      val_ind = np.append(np.random.choice(ind, NUM_VAL-len(val_ind)), val_ind)
  return label_dic, images, labels, val_ind

class ISIC18(Dataset):
  def __init__(self, root, train=True, transform=None):
    super(ISIC18, self).__init__()
    self.transform = transform
    self.train = train
    
    if self.train:
      self.data_folder = DATAPATH + 'Train/'
      self.classes, self.img_paths, self.labels, self.val_ind = getlabels('Train')
    else:
      self.data_folder = DATAPATH + 'Test/'
      self.classes, self.img_paths, self.labels, self.val_ind = getlabels('Test')

  def __getitem__(self, index):
    img_path = self.img_paths[int(index)]
    label = self.labels[int(index)]
    img = Image.open(self.data_folder + img_path + '.jpg')

    if self.transform is not None:
      img = self.transform(img)

    return img, label

  def __len__(self):
    return len(self.labels)

def getdata():

  print ("Collecting data ...")

  transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.7635, 0.5461, 0.5705), std=(0.6332, 0.3557, 0.3974))
  ])

  isic18_train = ISIC18(DATAPATH, train=True, transform=transform)
  sample = isic18_train.__getitem__(0)[0][None, :, :, :]
  print(sample.shape)
  train_dataloader = DataLoader(isic18_train, batch_size=1, sampler=sampler.SubsetRandomSampler(list(set(range(NUM_TRAIN+NUM_VAL)).difference(set(isic18_train.val_ind)))))

  isic18_val = ISIC18(DATAPATH, train=True, transform=transform)
  val_dataloader = DataLoader(isic18_val, batch_size=1, sampler=sampler.SubsetRandomSampler(isic18_val.val_ind))

  # isic18_test = ISIC18(DATAPATH, train=False, transform=transform)
  # test_dataloader = DataLoader(isic18_test, batch_size=64)
  test_dataloader = None

  print ("Collect data complete!\n")

  return (train_dataloader, val_dataloader, test_dataloader, sample)

def compute_mean():
  transform = T.Compose([
    # T.Resize((224,224)),
    T.ToTensor(),
  ])
  isic18_train = ISIC18(DATAPATH, train=True, transform=transform)
  img0 = isic18_train.__getitem__(0)[0][None, :, :, :]
  mean = torch.mean(img0, dim=(2,3))/isic18_train.__len__()
  for i in range(1, isic18_train.__len__()):
    img = isic18_train.__getitem__(i)[0][None, :, :, :]
    mean += torch.mean(img, dim=(2,3))/isic18_train.__len__()
    print(mean)
  print (mean.size(), mean)

def compute_var():
  mean = torch.tensor([0.7635, 0.5461, 0.5705])
  transform = T.Compose([
    # T.Resize((224,224)),
    T.ToTensor(),
  ])
  isic18_train = ISIC18(DATAPATH, train=True, transform=transform)
  img0 = isic18_train.__getitem__(0)[0][None, :, :, :]
  img = img0 - mean[None, :, None, None]
  img = img * img
  totalvar = torch.sum(img * img, dim=(2,3))/(450*600*isic18_train.__len__())
  for i in range(1, isic18_train.__len__()):
    img = isic18_train.__getitem__(i)[0][None, :, :, :]
    img = img * img
    totalvar += torch.sum(img * img, dim=(2,3))/(450*600*isic18_train.__len__())
    print (i, ':', totalvar)
  print (totalvar.size(), totalvar)

# 不进行resize，直接计算的norm参数
# mean = [0.7635, 0.5461, 0.5705]
# var = [0.4009, 0.1265, 0.1579]

