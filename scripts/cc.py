from PIL import Image
import os
import cv2
import sys
sys.path.append('/home/huihui/Project/DL-toolbox/')
import glob
from tools import colorConstancy

frompath = '/home/huihui/Data/ISIC2019/Data'
topath = '/home/huihui/Data/ISIC2019_resize2_cc/Data'

def getClasses():
  return glob.glob(frompath+'/*')

def makepath(outpath):
  if not os.path.exists(outpath):
      os.mkdir(outpath)

def readImg(inpath):
  # return Image.open(inpath)
  return cv2.imread(inpath)

def convert_cc2(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return colorConstancy.shades_of_gray(img)

def convert_cc(img):
  """
  Only color constancy.
  """
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(img)

def convert_resize(img):
  resize = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
  resize = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
  return resize

def convert_resize_cc(img):
  """
  Resize to 500 * 500 -> Color Constancy.
  """
  resize = cv2.resize(img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
  resize = cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(resize)

def convert_resize_crop_cc(img):
  """
  Resize with resolution holds -> Center Crop 500 * 500 -> Color Constancy.
  """
  h, w, c = img.shape
  scale = 500. / min(h, w)
  new_h = int(h*scale)+1
  new_w = int(w*scale)+1
  h_from = int((new_h-500)/2)
  w_from = int((new_w-500)/2)
  # print (img.shape)
  resize = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # print(resize.shape)
  crop = resize[h_from:(h_from+500), w_from:(w_from+500)]
  # print(crop.shape)
  res = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(res)

def convert_resize2_cc(img):
  h, w, c = img.shape
  scale = 500. / min(h, w)
  new_h = int(h*scale)
  new_w = int(w*scale)
  # print (img.shape)
  resize = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # print(resize.shape)
  # crop = resize[h_from:(h_from+500), w_from:(w_from+500)]
  # print(crop.shape)
  res = cv2.cvtColor(resize,cv2.COLOR_BGR2RGB)
  return colorConstancy.Grey_world(res)

def convert_resize_crop(img):
  """
  Resize with resolution holds -> Center Crop 500 * 500.
  """
  h, w, c = img.shape
  scale = 500. / min(h, w)
  new_h = int(h*scale)+1
  new_w = int(w*scale)+1
  h_from = int((new_h-500)/2)
  w_from = int((new_w-500)/2)
  # print (img.shape)
  resize = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # print(resize.shape)
  crop = resize[h_from:(h_from+500), w_from:(w_from+500)]
  # print(crop.shape)
  res = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
  return res

def saveImg(newImg, outpath):
  img = Image.fromarray(newImg)
  img.save(outpath)

def main():
  classes = getClasses()
  convert = convert_resize2_cc
  if not os.path.exists(topath):
    os.mkdir(topath)
  for i in range(len(classes)):
    c = classes[i]
    newc = c.replace(frompath, topath)
    makepath(newc)
    for imgpath in glob.glob(c+'/*.jpg'):
      print ('Converting {}'.format(imgpath))
      nimg = readImg(imgpath)
      newimg = convert(nimg)
      outpath = imgpath.replace(frompath, topath)
      saveImg(newimg, outpath)

main()