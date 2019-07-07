import pandas as pd
import numpy as np
import os


# * * * * * * * * *
# Modify the Data Path
# datapath
#   - groundtruth.csv
# * * * * * * * * *
datapath = '/home/huihui/Data/ISIC2018_openset'

def toimagefolder():
  """
  Turn the images into ImageFolder configuration
  """
  train = pd.read_csv(datapath+'/groundtruth.csv')

  print (train.head())

  classes = train.columns.to_list()[1:]
  C = len(classes)
  imgs = train['image']
  print (classes, C, imgs.shape)

  if not os.path.exists(datapath+'/Data'):
    os.system('mkdir {}/Data'.format(datapath))

  for v in range(C):
    mask = train[classes[v]] == 1
    imglist = imgs[mask].to_list()
    if not os.path.exists(datapath + '/Data/' + classes[v]):
      os.system('mkdir {}/Data/{}'.format(datapath, classes[v]))
    for k in range(len(imglist)):
      os.popen('cp {}/all/{}.jpg {}/Data/{}/'.format(datapath, imglist[k], datapath, classes[v]))
    print(imgs[mask].shape)
    
def get5folds():
  """
  Create validation set index and 
  """
  import torchvision.datasets as dset
  data = dset.ImageFolder(datapath+'/Data')
  # data = pd.read_csv(datapath+'/groundtruth.csv')
  classes = np.array(data.samples)[:, 1]
  C = np.unique(classes).shape[0]

  folds = [np.array([]) for i in range(5)]
  for i in range(C):
    # col = data[labels[i]]
    # mask = col == 1
    # imgs = data.index[mask].to_list()
    # np.random.shuffle(imgs)
    # c_folds = np.array_split(imgs, 5)
    # folds = [np.hstack((folds[i], c_folds[i])) for i in range(5)]
    inds = np.where(classes == '{}'.format(i))[0]
    np.random.shuffle(inds)
    c_folds = np.array_split(inds, 5)
    folds = [np.hstack((folds[i], c_folds[i])) for i in range(5)]
  print (len(folds))
  for _ in range(5):
    print(folds[_].shape)
    np.save('{}/{}fold.npy'.format(datapath, str(_)), folds[_])

  # np.save('/home/huihui/Data/ISIC2018/validation.npy', validation)
  # np.save('/home/huihui/Data/ISIC2018/train.npy', train)

get5folds()