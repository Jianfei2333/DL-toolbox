from PIL import Image
import os
import cv2
import sys
sys.path.append('/home/huihui/Project/DL-toolbox/')
import glob
from tools import colorConstancy

frompath = '/home/huihui/Data/ISIC2019/Data'
topath = '/home/huihui/Data/ISIC2019/resize_crop'

def getClasses():
  return glob.glob(frompath+'/*')

def makepath(outpath):
  if not os.path.exists(outpath):
      os.mkdir(outpath)

def readImg(inpath):
  # return Image.open(inpath)
  return cv2.imread(inpath)

def convert_cc(img):
  """
  Only color constancy.
  """
  return colorConstancy.Grey_world(img)


def convert_resize_cc(img):
  """
  Resize to 1024 * 1024 -> Color Constancy.
  """
  resize = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
  resize = cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(resize)

def convert_resize_crop_cc(img):
  """
  Resize with resolution holds -> Center Crop 1024 * 1024 -> Color Constancy.
  """
  h, w, c = img.shape
  scale = 1024. / min(h, w)
  new_h = int(h*scale)+1
  new_w = int(w*scale)+1
  h_from = int((new_h-1024)/2)
  w_from = int((new_w-1024)/2)
  # print (img.shape)
  resize = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # print(resize.shape)
  crop = resize[h_from:(h_from+1024), w_from:(w_from+1024)]
  # print(crop.shape)
  res = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(res)

def saveImg(newImg, outpath):
  img = Image.fromarray(newImg)
  img.save(outpath)

def main():
  classes = getClasses()
  if not os.path.exists(topath):
    os.mkdir(topath)
  for i in range(len(classes)):
    c = classes[i]
    newc = c.replace(frompath, topath)
    makepath(newc)
    for imgpath in glob.glob(c+'/*.jpg'):
      print ('Converting {}'.format(imgpath))
      nimg = readImg(imgpath)
      newimg = convert_resize_crop_cc(nimg)
      outpath = imgpath.replace(frompath, topath)
      saveImg(newimg, outpath)

main()