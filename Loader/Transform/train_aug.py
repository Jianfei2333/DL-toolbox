import numpy as np
import torchvision.transforms as T

def load(modelinfo, info):
  mean = info['mean']
  std = info['std']
  normalize = T.Normalize(mean=mean, std=std)

  inputsize = modelinfo['inputsize']
  resize = tuple([int(x*(4/3)) for x in inputsize])

  transform = {
    'train': T.Compose([
      T.Resize(resize), # 放大
      T.RandomResizedCrop(inputsize), # 随机裁剪后resize
      T.RandomHorizontalFlip(0.5), # 随机水平翻转
      T.RandomVerticalFlip(0.5), # 随机垂直翻转
      T.RandomApply([T.RandomRotation(90)], 0.5), # 随机旋转90/270度
      T.RandomApply([T.RandomRotation(180)], 0.25), # 随机旋转180度
      T.RandomApply([T.ColorJitter(brightness=np.random.random()/5+0.9)], 0.5), #随机调整图像亮度
      T.RandomApply([T.ColorJitter(contrast=np.random.random()/5+0.9)], 0.5), # 随机调整图像对比度
      T.RandomApply([T.ColorJitter(saturation=np.random.random()/5+0.9)], 0.5), # 随机调整图像饱和度
      T.ToTensor(),
      normalize
    ]), 
    'val': T.Compose([
      T.Resize(inputsize), # 放大
      T.ToTensor(),
      normalize
    ])
  }

  return transform