from PIL import Image
import os
import sys
sys.path.append('/home/huihui/Project/DL-toolbox/')
import glob
from tools import colorConstancy

frompath = '/home/huihui/Data/ISIC2019/Data'
topath = '/home/huihui/Data/ISIC2019/ColorConstancy'

def getClasses():
  return glob.glob(frompath+'/*')

def makepath(outpath):
  if not os.path.exists(outpath):
      os.mkdir(outpath)

def readImg(inpath):
  return Image.open(inpath)

def convert(img):
  return colorConstancy.Grey_world(img)

def saveImg(newImg, outpath):
  img = Image.fromarray(newImg)
  img.save(outpath)

def main():
  classes = getClasses()
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