import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

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

def test_ISIC2018():
  td1, vd, td2, sp = getdata()
  